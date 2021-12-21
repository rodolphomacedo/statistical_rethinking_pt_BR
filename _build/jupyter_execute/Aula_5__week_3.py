#!/usr/bin/env python
# coding: utf-8

# # As mutias variáveis e os waffles espúrios

# <img src="./images/waffle_house.jpeg" alt="Waffle House" width=1000  />
# 
# [Fonte](https://br.linkedin.com/company/waffle-house)

# <img src='waffle_house.jpg' alt='Walffle House'/>

# Nessas aulas iremos começar a construçcão de modelos de regressões múltiplas e começaremos a criar as bases para o framework de inferênias causais.
# 
# Para iniciarmos a discusão iremos introduzir um exemplo empírico, ou seja, um exemplo baseado na experiência e observações, sejam elas baseadas em algum método (*metódicas*) ou não.

# A *waffle house* é uma cadeia de restaurantes com mais de 2000 locais em 25 estados nos EUA (mapa amarelo abaixo). A maioria das suas localizações estão no Sul do país e é um item da cultura regional norte americana. (*[wikipedia](https://en.wikipedia.org/wiki/Waffle_House)*). Uma das particularidades dessa rede de restaurantes é que eles trabalham 24 horas. Nunca fecham! Essa é a proposta de negócio deles.
# 
# Um outro fato importante, e desagradável, que se surge no sul dos EUA são os *Furacões*, que são causados pelas depressões tropicais climáticas do pais. Os restaurantes da rede Waffles são um dos únicos estabelecimentos que continuam abertos quando ocorrem os periodos furacões. Exceto quando esses furacões atingem uma de suas lojas.
# 
# A rede é tão confiável que governadores dos EUA criaram o *waffle house index*, internamente na FEMA (*agência federal de gestão de emergências*), como uma métrica informal com o nome da rede de restaurantes para determinar o efeito de tempestades, com uma escala de ajuda necessária para a recuperação de um desastre. (*[waffle house index](https://en.wikipedia.org/wiki/Waffle_House_Index)*)

# <img src="./images/WH_per_people.png" alt="waffle house map" width=900 />
# 
# Imagem - https://www.scottareynhout.com/blog/2017/10/7/waffle-house-map

# Além dos desastres naturais, existem também muitas outras coisas acontecendo em grande escala no sul dos EUA, tal como **divórcios**!
# 
# No gráfico abaixo temos a indicação da quantidade de divórcios nos EUA. Observe o sul do país e compare com o mapa acima.

# <img src="./images/WH_per_divorce.png" alt="Waffle House contry-level-marriage-divorce-data-2010" width=900 />
# 
# Imagem - https://www.bgsu.edu/ncfmr/resources/data/original-data/county-level-marriage-divorce-data-2010.html

# Percebeu? Em ambos os mapas existem uma grande concentração no extremo do sul de restaurantes do Waffle House e subindo mais ao norte do país temos quantidades cada vez menores.
# O mesmo ocorre no mapa da taxa de divórcios, quando olhamos para o sul do país.
# 
# Podemos então fazer uma estimativa esses dados estão correlacionados entre si. E podemos então pensar que quanto maior a concetração de restaurantes na região maior seria seria a taxa de divórcios.
# 
# E por que isso acontece? Pela seguinte razão. 
# 
# Por nada! 
# 
# Sim nada!!!
# 
# Não existe nada que tenha uma relação direta que um quantidade de restaurantes em algum local influêncie casais a brigrarem e tomarem a decisão de se separar! 
# 
# É estranho. É cômico. É intrigante. É uma Correlação Espúria!

# ### Correlações Espúrias
# 
# Essas são correlações espúrias, ou seja, correlações sem certeza; que não é verdadeira nem real; é hipotética!
# 
# Muitas coisas estão relacionadas com as outras no mundo.
# 
# Isso é a Natureza!
# 
# Por exemplo, se quisermos, por qualquer motivo que seja, arrumar um argumento para enfraquecer a imagem da rede Waffle House, podemos usar essas corelações espúrias com um dos argumento. Assim, iriamos expor na mídia que `...estudos indicam que o aumento do número de restaurantes da rede Waffle House na região aumentou drasticamente o número de divórcios na região.`
# 
# ----
# 
# Soa estranho, eu sei! Mas, lá no fundo, esse tipo de pensamento não soa tão estranho no dia a dia...

# Existem diversas correlações espúrias no mundo. Muita coisa tem correlação com muitas outras coisas.
# 
# Entretanto, `essas correlações não implicam causalidade`.

# Mas para entendermos melhor, vamos ver mais alguns exemplos sobre essas correlações espúrias:
# 
# 
# - O consumo de queijo tem uma correlação de $94.7\%$ com os acidentes fatais com o emaranhado do lençol de cama. 
# 
# 
# - O acidentes de afogamentos em piscinas tem correlação de $66\%$ com o número de filmes do Nicolas Cage por ano.
# 
# 
# - O consumo per capito de frango tem uma correlação de $89\%$ com a importação de petróleo.
# 
# Percebeu?
# 
# Se o consumo de frango diminuisse, a importação provavelmente não sofreria nenhum impacto por essa causa. Se o Nicolas Cage se aposentar das telas, os acidentes por afogamento continuarão constantes. E, caso o consumo de queijo diminuir, também não haverão uma diminuição nos acidentes fatais das pessoas que estão dormindo em suas camas.
# 
# `Correlação não implica causalidade!`
# 
# O exemplo gráfico do Nicolas Cage.
# 
# <img src="./images/chart.jpeg" alt="Tyler Vigen spurious correlations" width=1000>
# 
# 
# Por fim, ter mais lojas da rede Waffle House não `causa` mais divórcios na região.
# 
# 
# 
# ----
# Mais correlações espúrias, tal como essas, podem ser vistas no site do [Tyler Vigen](https://www.tylervigen.com/spurious-correlations).

# Entendido essa parte, vamos ao objetivo desse capítulo.
# 
# Vamos ver como se faz um modelo de regressão linear novamente, mas agora iremos ver também como se faz com múltiplas variáveis e quais são suas implicações. 
# 
# `Modelo de Regressão Múltipla`:
# 
# - A parte boa desse tipo de modelos é que as regressões múltiplas podem revelar correlações espúrias e podem também revelar associações escondidas que nós não fariamos normalmente não teriamos visto usando o modelo com uma simples variável preditora.
# 
# 
# - Mas, por outro lado, podemos também adicionar variáveis explicativas aos modelos que contenham correlações espúrias à regressão múltipla e pode também esconder as algumas das reais associações.
# 
# 
# 
# Então, como essas coisas geralmente não são bem explicadas, vamos detalhar todo o processo de construção de uma regressão múltipla, a seguir.
# 
# Quando construímos um modelo usando uma regressão múltipla devemos ter um estrutura mais ampla para pensar sobre as nossas decisões. Apenas *jogar* todas as variáveis explicativas dentro da regressão múltipla, como usualemente é feito, é o segredo para o fracasso da análise e não queremos fazer isso!
# 
# Para isso precisamos de uma estrutura mais ampla para conseguirmos pensar e tomar melhores decisões. Essa estrutura é o framework de inferência casual.
# 
# `O que iremos aprender do básico sobre inferência casual`:
# 
# - Grafos acíclicos direcionados (DAGs - *Directed acyclic graphs*)
# 
# 
# - Fork, pipes, colliders...
# 
# 
# - Critério de Backdoor.

# Já sabemos que o Waffle House não causa os divórcios. Mas o que causa os divórcios?
# 
# Já vimos no mapa acima que divórcios do sul do país tem uma taxa bem mais alta do que no restante mais ao norte. Existem muitos esforços para tentar identificar as causas e as taxas de divórcios. Sabemos que no sul tem uma predominância religiosa quando comparada ao restante do país. Isso deixa os cientistas com certas desconfianças.

# Existem assim muitas coisas que estão correlacionadas com a taxa de divórcios. Uma delas é a taxa de casamentos. Tasmbém podemos usar para cada um dos outros 50 estados do país, todos eles tem uma correlação positiva da taxa de casamentos com a taxa de divórcios.
# 
# Um ponto importante nessa correlação que é que só pode acontecer um divórcio se houver um casamento! 
# 
# Mas a correlação entre essas taxas podem ser também `correlações espúrias`. Assim como a correlação entre homícidos e divórcios é uma correlação espúria.
# 
# Pois uma taxa alta de casamentos pode indicar que a sociedade vê o casamento de modo favorável e isso pode significar taxas de divórcios mais baixas. Não necessáriamente faz sentido, mas pode ser que as taxas de casamentos e de divórcios sejam correlações espúrias também. 
# 
# 
# - Correlação não impica Causalidade
# 
# 
# - Causalidade não implica Correlação
# 
# 
# - Causalidade implica correlação condicional
# 
# 
# - Precisamos mais do que apenas simples modelos
# 
# 
# O que causa o divórcio? 
# 
# Vamos descobrir isso...

# Existe outra variável que também é correlacionada com a variável `taxa de divórcio`, é a variável `idade mediana das pessoas que se casam nos estados`. Mas diferente da `taxa de casamento`, essa variável tem *correlação negativa*.

# In[1]:


import numpy as np
import pandas as pd
import stan
import nest_asyncio

import plotly.express as px  # Vamos usar Plotly ao invés do Matplolib
import plotly.graph_objects as go  # Usado para colocar múltiplos gráficos no plotly
from plotly.subplots import make_subplots  # Usado para gerar vários gráficos na mesma figura


# In[2]:


# Desbloqueio do asyncIO do jupyter
nest_asyncio.apply()


# In[3]:


# Lendo os dados

df = pd.read_csv('./data/WaffleDivorce.csv', sep=';')
df.head()


# In[4]:


model_stan_divorce = """
    data {
        int N;
        vector[N] tx_divorce;
        vector[N] tx_marriage;
        vector[N] tx_median_age;
    }

    parameters {
        real alpha;
        real beta_marriage;
        real beta_median_age;
        real<lower=0, upper=5> sigma;
    }
    
    model {
        alpha ~ normal(20, 3);
        beta_marriage ~ normal(0, 3);
        beta_median_age ~ normal(30, 6);
    
        tx_marriage ~ normal(alpha + beta_marriage + beta_median_age, sigma);
    }
"""

data = {
    'N': len(df),
    'tx_divorce': df.Divorce.values,
    'tx_marriage': df.Marriage.values,
    'tx_median_age': df.MedianAgeMarriage.values,
}

posteriori_divorce = stan.build(model_stan_divorce, data=data)
fit_divorce = posteriori_divorce.sample(num_chains=4, num_samples=1000)

posteriori_divorce_alpha = fit_divorce['alpha'].flatten()
posteriori_divorce_marriage = fit_divorce['beta_marriage'].flatten()
posteriori_divorce_age = fit_divorce['beta_median_age'].flatten()
posteriori_sigma = fit_divorce['sigma'].flatten()


# In[5]:


fig = px.histogram(posteriori_divorce_alpha, 
                   nbins=100,
                   histnorm='probability density', 
                   labels={'variable':'Posteriori Divorce'},
                   opacity=0.8,
                   marginal="box",  # Pode ser também `box` (box-plot), `violin` ou 'rug'
                  )
fig.update_layout(bargap=0.2, title_text="Distribuição Posteriori dos Divórcios", showlegend=False)
fig.show()


# In[6]:


fig = make_subplots(rows=1, cols=2)  # Inicia uma figura no plotly

trace_marriage = go.Histogram(x=posteriori_divorce_marriage,
                              xbins=dict(size=0.1),
                              name='Taxa de Casamentos',
                              opacity=0.5,
                              marker_color='blue',
                              histnorm='probability density')

trace_age = go.Histogram(x=posteriori_divorce_age,
                         xbins=dict(size=0.3),
                         histnorm='probability density',
                         name='Mediana das idades',
                         opacity=0.5,
                         marker_color='red')

fig.append_trace(trace_marriage, 1, 1)
fig.append_trace(trace_age, 1, 2)

fig.update_layout(
    title_text='Densidades das Posterioris dos Betas Marriage & Divorce',
    yaxis_title_text='Probabilidade á Posteriori',
    bargap=0.2
)

fig.show()


# In[7]:


divorce_marriage = []

range_divorces = np.linspace(df.Divorce.min(), df.Divorce.max() + 1, 100)

for divorce_i in range_divorces:
    divorce_marriage.append(posteriori_divorce_alpha + posteriori_divorce_marriage * divorce_i)
    
divorce_marriage = np.array(divorce_marriage).T

fig = go.Figure()
for i in range(100):
    fig.add_trace(go.Scatter(x=range_divorces, 
                             y=divorce_marriage[i],
                             mode='lines',
                             colors='blue'))

fig.show()


# In[ ]:


len(df.Divorce.values)


# In[ ]:


# fig = make_subplots(rows=1, cols=2)
fig = go.Figure()
fig.add_trace(go.Scatter(x=posteriori_divorce_alpha,
                         y=,
                         line_color='red'))


# 

# In[ ]:


# 8:42 Colocar o gráfico de correlação entre as taxas de casamentos e divórcios e também mediana das idades e das taxas de divórcios.


# Quais dessas correlações acimas implicam causalidade?
# 
