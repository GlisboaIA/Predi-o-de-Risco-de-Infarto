from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Estrutura da rede incluindo Arritmia
model = BayesianNetwork([
    ('Idade', 'RiscoInfarto'),
    ('HistoricoFamiliar', 'RiscoInfarto'),
    ('DorPeito', 'RiscoInfarto'),
    ('RiscoInfarto', 'Arritmia'),
    ('DorPeito', 'Arritmia')
])

# 2. CPDs originais
cpd_idade = TabularCPD('Idade', 3, [[0.2], [0.3], [0.5]], state_names={'Idade': ['<45', '45-60', '>60']})
cpd_historico = TabularCPD('HistoricoFamiliar', 2, [[0.2], [0.8]], state_names={'HistoricoFamiliar': ['Sim', 'Não']})
cpd_dor = TabularCPD('DorPeito', 2, [[0.7], [0.3]], state_names={'DorPeito': ['Sim', 'Não']})

cpd_risco = TabularCPD(
    'RiscoInfarto', 2,
    values=[
        [0.8, 0.6, 0.7, 0.4, 0.7, 0.5, 0.9, 0.8, 0.6, 0.3, 0.5, 0.2],  # Alto
        [0.2, 0.4, 0.3, 0.6, 0.3, 0.5, 0.1, 0.2, 0.4, 0.7, 0.5, 0.8]   # Baixo
    ],
    evidence=['Idade', 'HistoricoFamiliar', 'DorPeito'],
    evidence_card=[3, 2, 2],
    state_names={
        'RiscoInfarto': ['Alto', 'Baixo'],
        'Idade': ['<45', '45-60', '>60'],
        'HistoricoFamiliar': ['Sim', 'Não'],
        'DorPeito': ['Sim', 'Não']
    }
)

# 3. Nova CPD: Arritmia depende de RiscoInfarto e DorPeito
cpd_arritmia = TabularCPD(
    'Arritmia', 2,
    values=[
        # Arritmia = Sim
        [0.9, 0.6, 0.7, 0.2],
        # Arritmia = Não
        [0.1, 0.4, 0.3, 0.8]
    ],
    evidence=['RiscoInfarto', 'DorPeito'],
    evidence_card=[2, 2],
    state_names={
        'Arritmia': ['Sim', 'Não'],
        'RiscoInfarto': ['Alto', 'Baixo'],
        'DorPeito': ['Sim', 'Não']
    }
)

# 4. Adiciona CPDs ao modelo
model.add_cpds(cpd_idade, cpd_historico, cpd_dor, cpd_risco, cpd_arritmia)

# 5. Verifica modelo
assert model.check_model()

# 6. Inferência
infer = VariableElimination(model)

# Consulta 1: RiscoInfarto e Arritmia
# Consulta 1 separada
query_risco1 = infer.query(
    variables=['RiscoInfarto'],
    evidence={'Idade': '>60', 'HistoricoFamiliar': 'Sim', 'DorPeito': 'Sim'}
)

query_arritmia1 = infer.query(
    variables=['Arritmia'],
    evidence={'Idade': '>60', 'HistoricoFamiliar': 'Sim', 'DorPeito': 'Sim'}
)

print(f"Consulta 1:\nRisco de Infarto (Alto): {query_risco1.values[0]:.2f}")
print(f"Probabilidade de Arritmia (Sim): {query_arritmia1.values[0]:.2f}")


query_risco2 = infer.query(
    variables=['RiscoInfarto'],
    evidence={'Idade': '<45', 'HistoricoFamiliar': 'Não', 'DorPeito': 'Não'}
)

query_arritmia2 = infer.query(
    variables=['Arritmia'],
    evidence={'Idade': '<45', 'HistoricoFamiliar': 'Sim'}
)
print(f"Consulta 2:\nRisco de Infarto (Alto): {query_risco2.values[0]:.2f}")
print(f"\nConsulta 2:\nProbabilidade de Arritmia (Sim): {query_arritmia2.values[0]:.2f}")

