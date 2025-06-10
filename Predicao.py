import tkinter as tk
from tkinter import ttk, messagebox
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from itertools import product

model = BayesianNetwork([
    ('Idade', 'RiscoInfarto'),
    ('HistoricoFamiliar', 'RiscoInfarto'),
    ('DorPeito', 'RiscoInfarto'),
    ('Tabagismo', 'RiscoInfarto'),
    ('Estresse', 'RiscoInfarto'),
    ('Hipertensao', 'RiscoInfarto'),
    ('RiscoInfarto', 'Arritmia'),
    ('DorPeito', 'Arritmia')
])

# CPDs
cpd_idade = TabularCPD('Idade', 3, [[0.05], [0.25], [0.7]], state_names={'Idade': ['<45', '45-60', '>60']})
cpd_historico = TabularCPD('HistoricoFamiliar', 2, [[0.8], [0.2]], state_names={'HistoricoFamiliar': ['Sim', 'Não']})
cpd_dor = TabularCPD('DorPeito', 2, [[0.7], [0.3]], state_names={'DorPeito': ['Sim', 'Não']})
cpd_tabagismo = TabularCPD('Tabagismo', 2, [[0.7], [0.3]], state_names={'Tabagismo': ['Sim', 'Não']})
cpd_estresse = TabularCPD('Estresse', 2, [[0.7], [0.3]], state_names={'Estresse': ['Sim', 'Não']})
cpd_hipertensao = TabularCPD('Hipertensao', 2, [[0.43], [0.57]], state_names={'Hipertensao': ['Sim', 'Não']})

states = list(product([0, 1, 2], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]))
risk_alto, risk_baixo = [], []
for s in states:
    score = s.count(0)
    p_alto = min(0.1 + score * 0.15, 0.99)
    risk_alto.append(round(p_alto, 2))
    risk_baixo.append(round(1 - p_alto, 2))

cpd_risco = TabularCPD(
    variable='RiscoInfarto', variable_card=2,
    values=[risk_alto, risk_baixo],
    evidence=['Idade', 'HistoricoFamiliar', 'DorPeito', 'Tabagismo', 'Estresse', 'Hipertensao'],
    evidence_card=[3, 2, 2, 2, 2, 2],
    state_names={
        'RiscoInfarto': ['Alto', 'Baixo'],
        'Idade': ['<45', '45-60', '>60'],
        'HistoricoFamiliar': ['Sim', 'Não'],
        'DorPeito': ['Sim', 'Não'],
        'Tabagismo': ['Sim', 'Não'],
        'Estresse': ['Sim', 'Não'],
        'Hipertensao': ['Sim', 'Não']
    }
)

cpd_arritmia = TabularCPD(
    'Arritmia', 2,
    values=[[0.9, 0.6, 0.7, 0.2], [0.1, 0.4, 0.3, 0.8]],
    evidence=['RiscoInfarto', 'DorPeito'],
    evidence_card=[2, 2],
    state_names={
        'Arritmia': ['Sim', 'Não'],
        'RiscoInfarto': ['Alto', 'Baixo'],
        'DorPeito': ['Sim', 'Não']
    }
)

model.add_cpds(
    cpd_idade, cpd_historico, cpd_dor,
    cpd_tabagismo, cpd_estresse, cpd_hipertensao,
    cpd_risco, cpd_arritmia
)

model.check_model()
infer = VariableElimination(model)

# --- INTERFACE TKINTER ---
root = tk.Tk()
root.title("Predição de Infarto e Arritmia")
root.geometry("500x400")
root.resizable(False, False)

frame = ttk.Frame(root, padding=20)
frame.pack()


campos = {}
valores = {
    'Idade': ['<45', '45-60', '>60'],
    'HistoricoFamiliar': ['Sim', 'Não'],
    'DorPeito': ['Sim', 'Não'],
    'Tabagismo': ['Sim', 'Não'],
    'Estresse': ['Sim', 'Não'],
    'Hipertensao': ['Sim', 'Não']
}

for idx, (variavel, opcoes) in enumerate(valores.items()):
    ttk.Label(frame, text=f"{variavel}:").grid(row=idx, column=0, sticky="w")
    var = tk.StringVar(value=opcoes[0])
    box = ttk.Combobox(frame, textvariable=var, values=opcoes, state="readonly", width=10)
    box.grid(row=idx, column=1, pady=5, padx=10)
    campos[variavel] = var

# Resultado
resultado = tk.Text(frame, height=5, width=40)
resultado.grid(row=len(valores) + 1, column=0, columnspan=2, pady=10)

def calcular():
    evidencias = {k: v.get() for k, v in campos.items()}
    try:
        risco = infer.query(['RiscoInfarto'], evidence=evidencias)
        arritmia = infer.query(['Arritmia'], evidence=evidencias)
        resultado.delete(1.0, tk.END)
        resultado.insert(tk.END, f"Risco de Infarto (Alto): {risco.values[0]:.2%}\n")
        resultado.insert(tk.END, f"Probabilidade de Arritmia (Sim): {arritmia.values[0]:.2%}")
    except Exception as e:
        messagebox.showerror("Erro", str(e))

# Botão
ttk.Button(frame, text="Calcular", command=calcular).grid(row=len(valores), column=0, columnspan=2, pady=10)

root.mainloop()
