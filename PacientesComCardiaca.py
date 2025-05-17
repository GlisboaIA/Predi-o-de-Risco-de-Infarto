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


cpd_idade = TabularCPD('Idade', 3, [[0.1], [0.2], [0.7]], state_names={'Idade': ['<45', '45-60', '>60']})
cpd_historico = TabularCPD('HistoricoFamiliar', 2, [[0.8], [0.2]], state_names={'HistoricoFamiliar': ['Sim', 'Não']})
cpd_dor = TabularCPD('DorPeito', 2, [[0.7], [0.3]], state_names={'DorPeito': ['Sim', 'Não']})
cpd_tabagismo = TabularCPD('Tabagismo', 2, [[0.7], [0.3]], state_names={'Tabagismo': ['Sim', 'Não']})
cpd_estresse = TabularCPD('Estresse', 2, [[0.7], [0.3]], state_names={'Estresse': ['Sim', 'Não']})
cpd_hipertensao = TabularCPD('Hipertensao', 2, [[0.43], [0.57]], state_names={'Hipertensao': ['Sim', 'Não']})


states = list(product([0, 1, 2], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]))  # 96 combinações

risk_alto = []
risk_baixo = []
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
    values=[
        [0.9, 0.6, 0.7, 0.2],  # Sim
        [0.1, 0.4, 0.3, 0.8]   # Não
    ],
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

assert model.check_model()


infer = VariableElimination(model)

pacientes = [
    ("Paciente 1", {'Idade': '>60', 'HistoricoFamiliar': 'Sim', 'DorPeito': 'Sim', 'Tabagismo': 'Sim', 'Estresse': 'Sim', 'Hipertensao': 'Sim'}),
    ("Paciente 2", {'Idade': '<45', 'HistoricoFamiliar': 'Não', 'DorPeito': 'Não', 'Tabagismo': 'Não', 'Estresse': 'Não', 'Hipertensao': 'Não'}),
    ("Paciente 3", {'Idade': '45-60', 'HistoricoFamiliar': 'Não', 'DorPeito': 'Não', 'Tabagismo': 'Não', 'Estresse': 'Não', 'Hipertensao': 'Não'}),
    ("Paciente 4", {'Idade': '>60', 'HistoricoFamiliar': 'Não', 'DorPeito': 'Não', 'Tabagismo': 'Sim', 'Estresse': 'Sim', 'Hipertensao': 'Sim'}),
    ("Paciente 5", {'Idade': '>60', 'HistoricoFamiliar': 'Sim', 'DorPeito': 'Sim', 'Tabagismo': 'Sim', 'Estresse': 'Sim', 'Hipertensao': 'Sim'}),
    ("Paciente 6", {'Idade': '45-60', 'HistoricoFamiliar': 'Sim', 'DorPeito': 'Sim', 'Tabagismo': 'Não', 'Estresse': 'Sim', 'Hipertensao': 'Sim'}),
    ("Paciente 7", {'Idade': '<45', 'HistoricoFamiliar': 'Não', 'DorPeito': 'Sim', 'Tabagismo': 'Sim', 'Estresse': 'Não', 'Hipertensao': 'Não'})
]


for nome, evidencias in pacientes:
    risco = infer.query(variables=['RiscoInfarto'], evidence=evidencias)
    arritmia = infer.query(variables=['Arritmia'], evidence=evidencias)
    print(f"\n{nome}")
    print(f"=> Risco de Infarto (Alto): {risco.values[0]:.2f}")
    print(f"=> Probabilidade de Arritmia (Sim): {arritmia.values[0]:.2f}")
