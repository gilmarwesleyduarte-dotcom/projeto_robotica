# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CÓDIGO FINAL: "GRÁFICO PERFEITO E TOQUE NA BOLA"
- Posiciona a bola automaticamente.
- Executa trajetória ultra-suave (10s).
- Gera gráfico de alta precisão.
"""

import sys
import tempfile
import math

# --- IMPORTAÇÕES ---
import matplotlib.pyplot as plt
import numpy as np
from controller import Supervisor

try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('O módulo "ikpy" não está instalado.')
# --- FUNÇÃO DE GRÁFICOS AVANÇADA (Cole isso no topo, após os imports) ---
def gerar_graficos_avancados(tempo, pos_ideal, pos_real, modo):
    """
    Gera 3 gráficos (Posição, Velocidade, Aceleração) com estilo técnico.
    Calcula as derivadas automaticamente para qualquer perfil.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Converter listas para arrays numpy
    t = np.array(tempo)
    x_ideal = np.array(pos_ideal)
    x_real = np.array(pos_real)

    # 2. Calcular Derivadas (Velocidade e Aceleração)
    # Ideal
    v_ideal = np.gradient(x_ideal, t)
    a_ideal = np.gradient(v_ideal, t)
    # Real
    v_real = np.gradient(x_real, t)
    a_real = np.gradient(v_real, t)

    # 3. Configurar Plotagem
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    cor_real = '#D32F2F' # Vermelho Técnico
    
    # Função auxiliar para estilo "Diagrama de Livro"
    def estilo_eixo(ax, titulo, ylabel):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_title(titulo, y=1.02, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, loc='top', rotation=0, fontsize=12, labelpad=-10)
        ax.set_xlabel('Tempo (t)', loc='right', fontsize=10, style='italic')
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- GRÁFICO 1: POSIÇÃO (s) ---
    estilo_eixo(ax1, f"Posição ({modo})", "s")
    ax1.plot(t, x_ideal, 'k--', linewidth=1.5, alpha=0.7, label='Planejado')
    ax1.plot(t, x_real, color=cor_real, linewidth=2, label='Real')
    ax1.legend(loc='best', fontsize=8)

    # --- GRÁFICO 2: VELOCIDADE (v) ---
    estilo_eixo(ax2, "Velocidade", "v")
    ax2.plot(t, v_ideal, 'k--', linewidth=1.5, alpha=0.7)
    ax2.plot(t, v_real, color='#E65100', linewidth=1.5, alpha=0.8) # Laranja

    # --- GRÁFICO 3: ACELERAÇÃO (a) ---
    estilo_eixo(ax3, "Aceleração", "a")
    ax3.plot(t, a_ideal, 'k--', linewidth=1.5, alpha=0.7)
    ax3.plot(t, a_real, color='#7B1FA2', linewidth=1, alpha=0.6) # Roxo (muito ruído)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# ⚙️ CONFIGURAÇÕES PARA O GRÁFICO PERFEITO
# ==============================================================================

# MODO: 'QUINTIC' é o mais suave matematicamente (curva em S perfeita).
TRAJECTORY_MODE = 'TRAPEZOIDAL' 

# DURAÇÃO: 10.0 segundos. 
# Motivo: Movimentos lentos permitem que o motor real acompanhe o planejamento sem erro.
TRAJ_DURATION = 5.0              

# OFFSET Z: Ajuste fino para Tocar a Bola sem Bater na Mesa.
# A bola tem raio de ~0.04. O centro é 0.05.
# Um offset de +0.02 mira no topo da bola. 
Z_OFFSET = 0.02 

# PRECISÃO: 50 tentativas por passo (Máxima precisão matemática).
IKPY_MAX_ITERATIONS = 50 
# ==============================================================================

if ikpy.__version__[0] < '3':
    sys.exit('A versão do módulo "ikpy" é muito antiga.')

# --- FUNÇÕES DE INTERPOLAÇÃO ---

def linear_interpolation(t, T, p0, pf):
    """Retorna a posição interpolada no tempo t (Linear)."""
    if t >= T: return pf
    return p0 + (t / T) * (pf - p0)

def cubic_interpolation(t, T, p0, pf):
    """Retorna a posição interpolada no tempo t (Cúbica)."""
    if t >= T: return pf
    a0 = p0
    a2 = (3 / (T**2)) * (pf - p0)
    a3 = (-2 / (T**3)) * (pf - p0)
    return a0 + a2 * (t**2) + a3 * (t**3)

def quintic_interpolation(t, T, p0, pf):
    """Retorna a posição interpolada no tempo t (Quíntica - A mais suave)."""
    if t >= T: return pf 
    a0 = p0
    a3 = (10 / (T**3)) * (pf - p0)
    a4 = (-15 / (T**4)) * (pf - p0)
    a5 = (6 / (T**5)) * (pf - p0)
    return a0 + a3 * (t**3) + a4 * (t**4) + a5 * (t**5)

def trapezoidal_interpolation(t, T, p0, pf, A_max=0.5):
    """Calcula a posição usando o Perfil de Velocidade Trapezoidal."""
    if t >= T: return pf
    if t <= 0: return p0
    delta_q = pf - p0
    abs_delta_q = abs(delta_q)
    
    if T < math.sqrt(4 * abs_delta_q / A_max): 
        Ta = T / 2.0
        V_max = abs_delta_q / Ta
        A = V_max / Ta
    else:
        Ta = (T - math.sqrt(T**2 - 4 * abs_delta_q / A_max)) / 2.0
        V_max = A_max * Ta
        A = A_max
        
    T1 = Ta 
    T2 = T - Ta 
    sign = 1 if delta_q > 0 else -1
    
    if t <= T1:
        q_t = p0 + sign * 0.5 * A * t**2
    elif t <= T2:
        q_t = p0 + sign * (0.5 * A * T1**2 + V_max * (t - T1))
    else:
        q_T2 = p0 + sign * (0.5 * A * T1**2 + V_max * (T2 - T1))
        t_prime = t - T2
        q_t = q_T2 + sign * (V_max * t_prime - 0.5 * A * t_prime**2)
    return q_t

# --- INICIALIZAÇÃO DO WEBOTS ---

supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

# Carrega URDF e Cadeia Cinemática
urdf_filename = supervisor.getUrdf()
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    file.write(urdf_filename.encode('utf-8'))
armChain = Chain.from_urdf_file(file.name, active_links_mask=[False, True, True, True, True, True, True, False])

# Inicializar motores
motors = []
for link in armChain.links:
    if 'motor' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(1.0) # Velocidade máxima dos motores
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timeStep)
        motors.append(motor)

target = supervisor.getFromDef('TARGET')
arm = supervisor.getSelf()


# Variáveis de Estado
traj_time = 0.0  
start_pos = [0, 0, 0] 
target_final_rel = [0, 0, 0]
INITIALIZED = False

# Listas para Gráfico
time_list = []
target_x_list = []
real_x_list = []
plot_shown = False

# Seleção de Função
if TRAJECTORY_MODE == 'QUINTIC':
    interp_func = quintic_interpolation
elif TRAJECTORY_MODE == 'TRAPEZOIDAL':
    interp_func = trapezoidal_interpolation
elif TRAJECTORY_MODE == 'LINEAR':
    interp_func = linear_interpolation
else:
    interp_func = cubic_interpolation

print(f'>>> INICIANDO TRAJETÓRIA {TRAJECTORY_MODE} <<<')
supervisor.getDevice('pen').write(False)

# --- LOOP ÚNICO ---
while supervisor.step(timeStep) != -1:
    
    # 1. SETUP INICIAL (Executa uma vez)
    if not INITIALIZED:
        # Lê onde o robô está AGORA (P0)
        current_joints = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
        fk_res = armChain.forward_kinematics(current_joints)
        start_pos = [fk_res[0, 3], fk_res[1, 3], fk_res[2, 3]]
        
        # Lê onde a bola está AGORA (Pf)
        targetPosition = target.getPosition()
        armPosition = arm.getPosition()
        
        # Define o alvo final relativo
        target_final_rel = [
            -(targetPosition[1] - armPosition[1]),
            targetPosition[0] - armPosition[0],
            (targetPosition[2] - armPosition[2]) + Z_OFFSET 
        ]
        
        INITIALIZED = True

    # 2. CÁLCULO DA TRAJETÓRIA (Durante 10s)
    if traj_time <= TRAJ_DURATION:
        
        # Calcula o ponto intermediário suave
        x = interp_func(traj_time, TRAJ_DURATION, start_pos[0], target_final_rel[0])
        y = interp_func(traj_time, TRAJ_DURATION, start_pos[1], target_final_rel[1])
        z = interp_func(traj_time, TRAJ_DURATION, start_pos[2], target_final_rel[2])
        
        traj_time += timeStep / 1000.0
        
        # Cinemática Inversa (Alta Precisão)
        initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
        ikResults = armChain.inverse_kinematics([x, y, z], max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)
        
        # Move Motores
        for i in range(len(motors)):
            motors[i].setPosition(ikResults[i + 1])

        # COLETA DADOS
        current_joints_read = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
        actual_pos = armChain.forward_kinematics(current_joints_read)
        
        time_list.append(traj_time)
        target_x_list.append(x)
        real_x_list.append(actual_pos[0][3])

# 3. GERAÇÃO DO GRÁFICO (Ao acabar)
    elif not plot_shown:
        print(f"Trajetória finalizada. Gerando painel completo para: {TRAJECTORY_MODE}...")
        
        # CHAMA A NOVA FUNÇÃO
        gerar_graficos_avancados(time_list, target_x_list, real_x_list, TRAJECTORY_MODE)
        
        plot_shown = True
        supervisor.simulationSetMode(0)