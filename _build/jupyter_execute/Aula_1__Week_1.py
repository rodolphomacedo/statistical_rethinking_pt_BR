#!/usr/bin/env python
# coding: utf-8

# # Statistical Rethinking

# <img src="./images/cover_book_SR2.jpg" alt="Statistical Rethinking 2" width=500>

# ## Antelóquio
# 
# 
# Todas as notas desse material foram baseadas nas aulas de 2019 ([ACESSE](https://www.youtube.com/playlist?list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI)) do curso do [`Prof. Richard McElreath`](https://www.mpg.de/9349950/evolutionary-anthropology-mcelreath) e também em seu livro (*capa acima*) Statistical Rethinking - *Segunda Edição*. 
# 
# Esse material foi construído tentando preservar os detalhes mais importantes e as construções manuais dos modelos na medida do possível e modo de ensino, permitindo assim um entendimento mais concreto do que está acontencendo.
# 
# Diferenças do material original: Esse material foi construído transportando as idéias codificadas em [R](https://cran.r-project.org/) do pacote [rethinking](https://github.com/rmcelreath/rethinking) para a linguagem [python](https://www.python.org/) em conjunto com suas principais bibliotecas de análise de dados (numpy, scipy, matplotlib) e no lugar do pacote *rethinking* usaremos, tanto quanto meu conhecimento alcançar, utilização da biblioteca [Stan](http://mc-stan.org/) e sua interface para python [pystan](https://pystan.readthedocs.io/en/latest/).
# 
# -----
# Esse material foi escrtio em lingua portuguesa (Brasil).
# 
# Estará disponibilizado gratuitamente no github: https://github.com/rodolphomacedo 
# 
# Erros, sugestões ou dúvidas podem ser enviadas para o email [rodolpho.ime@gmail.com](rodolpho.ime@gmail.com)
# 
# *Bom divertimento!!!*

# ## Prefácio
# 
# Esse material tem como objetivo primário descrever um modo de (re)pensar a estatística. Para uma primeira leitura não é necessário um foco mais atento dos códigos que geram os gráficos, porém o próprio código e os comentários são partes integrantes do método que usei para apresentar as idéias e muitas dúvidas podem sanadas com uma leitura mais atenta.
# 
# Tentei colocar o máximo de figuras e mêmes que pude encontrar para ilustrar os exemplos.

# ## Requisitos
# 
# É necessário um breve conhecimento da `teoria das probabilidades` e uma vivência razoável em python e suas bibliotecas mais usuais de análise de dados.

# 
# 
# ---
# ##  Golem de Praga
# 
# 
# <img src="./images/golem_of_prague.png" width=100% height=100%>

# Todos que são cientistas precisam fazer inferências sobre alguns aspectos de um subconjunto de particularidades do corpo de um problema que está sob estudo. 
# 
# Medir a Natureza é fascinante! Tal medida nos permite um entendimento aproximado de um subconjunto do funcionamento da maquinaria natural do Todo.
# 
# Ao se observar o fenômeno a ser estudado, a proposta da criação de uma métrica deriva-se de uma ideia, uma sugestão, uma proposta, que se admite, independentemente do fato de ser verdadeira ou falsa, como um princípio a partir do qual se pode deduzir um determinado conjunto de consequências. Tal proposta definiremos como a 
# `hipótese`.  
# 
# A realização de uma operação *intelectual*, por meio da qual se afirma a verdade de uma hipótese em decorrência de sua ligação com outras já reconhecidas como verdadeiras, isto é, `inferência`.

# Problemas de escopo aberto apresentam relativas dificuldades que desafiam a capacidade de nosso espírito em obtermos uma solução. Tais problemas contém uma particularidade de serem dificíes de obter conhecimento a seu respeito.   (**complementar com LAPLACE "sobre a probablidade" ***)

# A estatítica sugerida pelo biólogo Ronald Fisher, no início da década de 20, não contém em seu propósito a capacidade de resolver problemas de caráter tão amplo.  **(complementar com SHARON, capítulo3 em Fisher)**
# 
# Tais técnicas propostas, e estudadas ainda hoje, podem ser visto como *pequenos robôs* que precisam de uma entrada e produzem uma saída agnóstica a seu propósito. *Robôs* de modo geral são assim, são bons para realizar tarefas que suprem e extende a necessidade humana. 

# O ambiente é, de modo geral, extremamente confuso para entede-lô completamente, assim a construção de uma estrutura nos permitirá obter ao menos alguns pontos de sabedoria do sistema.

# ----

# ### Falha da Falsificação 
# ##### Karl Popper - ($25:51$)

# Karl Popper, um dos mais conhecidos filósofos da ciência, entra para a história por propors uma definição do que é `ciência` e, o que ela não é, através da falsificação ou não.
# 
# (Pode ser que o rítmo que estou fazendo todas as coisas na minha vida, seja realmente um rítmo bom pra MIM. E prestar atenção na **pressa de terminar tudo ontem** quando me pegar assim, tentar diminuir até a serenidade, mas a questão é como posso melhorar a ansiedade, a qualidade da escrita, e principalmente como criar um sistema de ação pelo qual consiga caminhar pelas pelas melhores estradas de todas as possíveis em todas as coisas, (*Ler os Jardim que se birfucam*))
# A m. abre uma capacidade de pensar de forma mais ampla, tem que perceber quando estiver fechando, o que estou voltando a ter. Daí treinar para reverter sozinho.)
# 
# O critério de falsificação é a demarcação do que está acontecendo dentro e fora. Mas há muitas outras coisas sobre as evidências que nos exige que tenhamos mais de um modelo para verificar quais deles são consistentemente com o que observamos. O que queremos é tentar falsificar é o modelo explicativo e não qualquer outro modelo sem importância.
# Agora no século XX, isso foi revertido, oque os cientistas tentam falsificar com seus testes estatísticos não são suas hipóteses de pesquisa, mas algumas hipóteses que eles não gostam e que nada está acontecendo. 
# 
# O que deveriamos realmente fazer é tentar fazer as previsões sobre o que está acontecendo e falsificar o restante.
# 
#  - *Assim, ciência não se trata de falsificar coisas, é necessário construir uma teoria substantiva em algum ponto.*
# 
# ### O que Karl Popper propõe:
# 
# *Construa uma hipótese de pesquisa substantiva com previsões pontuais sobre o que deveria estar acontecendo e tente falsifica-lá.*
# 
# **E não falsificar a ideia boba que não está acontecendo nada. Porque sempre está acontecendo algo, essa é a Natureza. Muitas as coisas estão correlacionadas em muitos lugares na natureza!**
# 
# A questão principal é:
# - **Como prever sua estrutura?**

# - Modelos *nulos* não são únicos!
# - Deveria *falsificar* o modelo explanatório, e não o modelo *nulo* ($h_0$).
# - Falsificação é consensual, não logico!
# - Falseabilidade é sobre a demarcação e não sobre o método.
# - Não exite um procedimento estatístico suficiente.
# - Ciência é uma tecnologia social.

# ----

# ### Engeharia de Golem

# Para o desenvolvimento dos golem's vamos precisar de um conjunto de princípios para que possamos construir nosso modelo estatístico. Não vamos entrar nesse curso achando que é apenas uma escolha de golem de dentro de uma caixa de ferramentas pré-fabricado.
# 
# Iremos construir os nosso próprios golems e também aprederemos os princícios para conseguirmos criticá-los e refiná-los.
# 
# Existem muitos modos de fazer essa escolha sob o direcionamento de filosofia que escolher trabalhar!
# 
# Nós iremos seguir esses três principios:
#     
# - Análise de dados Bayesiana
#     
# - Modelos multi-níveis
#     
# - Comparação de modelos
#     

# #### Análise de Dados Bayesiano

# - Contar todas as maneiras que os dado podem acontecer, de acordo com a suposição.
# 
# - As suposições que são mais consistentes com os dados, ou seja, as que ocorrem mais vezes, são mais plausíveis de acontecerem.
# 
# **Faremos algumas suposições sobre como o mundo poderá ser e também como é o processo casual acontece. Então iremos ver algumas observações que são consequencias desse processo.**
# 
# Assim podemos dizer que temos um conjunto de suposições alternativos, e cada uma dessas suposições são mais ou menos plauíveis de ocorrer de acordo com quantidade de vezes que ela já ocorreu.
# 
# **Isso é na verdade uma forma muito específica de contar as coisas.** Isso é o que o nossos Golems irão fazer, contar de forma  MUITO rápida! Nós precisamos apenas programá-los para que ele conte do modo que nós quizermos.

# ### Modelos Multiníveis
# 
# - Modelos com multiplos níveis de incerteza:
#     - Troque os parâmetros por modelos
#     
# 
# - Casos comuns de uso:
#     - Amostragem repetida e desbalanceada
#     - Estudo da variação
#     - Evitar a média.
#     - Filogenética, fator e análises, networks, modelos espaciais.
# 
# 
# - Estratégia Bayesiana Natural:
#     - Estratégia natural para a construção desses modelos.

# ### Comparação de modelos
# 
# 
# - Temos que ter múltiplos modelos para podermos compará-los e saber o que está acontecendo, e não falsificar uma hipótese nula. Estamos comparando o significado dos modelos.
# 
# 
# - Problemas básicos:
#     - Overfitting
#     - Inferência Casual (*Para descobrir alguma rede de causas e efeitos, pensar em uma rede de mediação*).
#     
#     
# - Navalha de Ocam é bobagem:
# 
# 
# - Teoria da Informação é menos bobagem:
#     - AIC, WAIC, cross-validation...
# 
# 
# - Devemos distinguir a predicação da inferência:
# 
#  
# 

# ---

# ### Erro de Colombo

# Colombo, enquanto navegava, enxergaria quais das direções que ele poderia seguir guiando-se apenas por seu mapa. O mapa não é o mundo real, é apenas uma representação, uma hipótese, do que ele pode ser. O mundo real é sempre muito maior e bem mais complexo. 
# 
# Nós, enquanto estamos construíndo um modelo, estamos nos guiando por um mapa lógico mental. Esse mapa também não é o mundo real, é apenas uma representação, uma hipótese, ou ainda uma sugestão particular, do que o mundo pode ser. 
# 
# * *O mundo real é sempre bem maior e bem mais complexo do que possa me parecer!*

# <img src="./images/Martin_Behaim_1492_Ocean_Map.png" width=300 height=300>
# 
# Cadê a América?
# 
# [Behain Globe](https://pt.wikipedia.org/wiki/Erdapfel): *Globo que Colombo usou viajando em direção as Américas.*
# 
# ---

# #### Sensu L.J. Savage (1954)
# 
# * **Pequeno Mundo**: Esse é o mundo das suposições dos Golems. Golems bayesianos são ótimos em um mundo pequeno.
# 
# * **Mundo Real**: Não existe a garantia de otimalidade para qualquer tipo de Golem.
# 
# Temos que nos preocupar com ambos!
# 
# 
# ----
# **Sobre Savage**:
#  - O economista Milton Friedman disse que Savage foi "*...uma das poucas pessoas que conheci a quem sem hesitar chamaria de gênio.*"
#  
# - Durante a Segunda Guerra Mundial, Savage serviu como principal assistente "estatístico" de John von Neumann.
# 
# [*Savage - Wikipedia*](https://en.wikipedia.org/wiki/Leonard_Jimmie_Savage)

# ### Jardim da Bifurcação dos Dados

# Para entender com funciona máquina bayesiana, vamos introduzir um exemplo simples:
# 
# *Obs: Ler o livro Borges, especificamente El jardim de sendeiros que se bifurcam*.
# 
# Temos uma bolsa com $4$ bolas ($N=4$). Sabemos que só existem a bolas *azuis* ($1$) e bolas *brancas* ($0$).
# 
# O que queremos saber é a quais são as $4$ bolas que tem dentro da bolsa? 
# 
# Vamos sortear as bolas e em seguida iremos devolver elas para a bolsa novamente. 
# 
# Em três sorteios, tivemos o seguinte resultado:
# 
# <img src="./images/blue_white_balls_exe.jpg"/>
# 
# 
# $$[Azul, Branca, Azul]$$
# 
# ou ainda, matematicamente:
# 
# $$[1, 0, 1] $$
# 
# Vamos construir um procedimento para obter mais conhecimento a respeito de quais são as bolas que estão na bolsa.

# A primeira coisa a se fazer é, listar todas as possíveis sugestões que pode acontecer, e elas são:
# 
# (1) $ [B, B, B, B] $
# 
# 
# (2) $ [A, B, B, B] $
# 
# 
# (3) $ [A, A, B, B] $
# 
# 
# (4) $ [A, A, A, B] $
# 
# 
# (5) $ [A, A, A, A] $

# As nossas 3 retiradas:
# 
# $[A, B, A] <=> [1, 0, 1]$

# In[1]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# In[2]:


### Jardim da Bifurcação dos Dados

N = 4  # Quantidade de bolas na bolsa.

amostra = [1, 0, 1]  # Nossas retiradas.


# In[3]:


### Contando quantas vezes cada hipótese pode ter ocorrido, dado nossa amostra.

plausibilidade_da_hipotese = {}  # Inicializando um novo dicionário com as hipóteses propostas.

                                           # ========= #
                                           #  AMOSTRA  #
                                           # ========= #
                                           # A - B - A #
                                           # ========= #
                    
plausibilidade_da_hipotese['B B B B'] = 0  # 0 * 4 * 0  = Nenhuma configuração é possível.
plausibilidade_da_hipotese['A B B B'] = 3  # 1 * 3 * 1  = 3
plausibilidade_da_hipotese['A A B B'] = 8  # 2 * 2 * 2  = 8
plausibilidade_da_hipotese['A A A B'] = 9  # 3 * 1 * 3  = 9
plausibilidade_da_hipotese['A A A A'] = 0  # 4 * 0 * 4  = Nenhuma configuração é possível.


# ----
# 
# *Esse o cerne da estatística bayesiana, contagens. Apenas contagens!*

# Uma das várias coisas interessantes que podemos fazer é usar a contagem anterior (*à priori*), e somar com a contagem atual:
# 
# Vamos, dessa vez, retirar mais uma bola ... e dessa vez, temos uma bola `Azul`:
# 
# $$[A]$$
# 

# In[4]:


# Quantidade de maneiras que podemos tirar uma bola Azul para cada hipótese(bolsa) proposta, 
#  ou seja, em cada bolsa hipotética.

nova_plausibilidade_da_hipotese = {}  # Inicializando um novo dicionário com as novas hipóteses.

nova_plausibilidade_da_hipotese['B B B B'] = 0  # 0  = Nenhuma configuraçao possível
nova_plausibilidade_da_hipotese['A B B B'] = 1  # 1 
nova_plausibilidade_da_hipotese['A A B B'] = 2  # 2
nova_plausibilidade_da_hipotese['A A A B'] = 3  # 3
nova_plausibilidade_da_hipotese['A A A A'] = 4  # 4 


# Assim teremos a contagem anterior (*à priori*) multiplicada pela nova retirada da bola Azul.

# In[5]:


# Atualizando a contagem com a nova informação (nova bola Azul)
# multiplicação é apenas uma forma de somar as possibilidades de cada hipótese.

plausibilidade = {}

plausibilidade['B B B B'] = plausibilidade_da_hipotese['B B B B'] * nova_plausibilidade_da_hipotese['B B B B']
plausibilidade['A B B B'] = plausibilidade_da_hipotese['A B B B'] * nova_plausibilidade_da_hipotese['A B B B']
plausibilidade['A A B B'] = plausibilidade_da_hipotese['A A B B'] * nova_plausibilidade_da_hipotese['A A B B']
plausibilidade['A A A B'] = plausibilidade_da_hipotese['A A A B'] * nova_plausibilidade_da_hipotese['A A A B']
plausibilidade['A A A A'] = plausibilidade_da_hipotese['A A A A'] * nova_plausibilidade_da_hipotese['A A A A']


# In[6]:


# Mostrando, para cada uma das hipóteses, o quão plausível é cada uma delas. 

print('Número de manerias diferentes de conseguir obter essa amostra, dado a hipótese atual. \n')

print('Plausibildade da hipótese [B B B B] = ', plausibilidade['B B B B'], 'maneiras possíveis, dado a hipótese.')
print('Plausibildade da hipótese [A B B B] = ', plausibilidade['A B B B'], 'maneiras possíveis, dado a hipótese.')
print('Plausibildade da hipótese [A A B B] = ', plausibilidade['A A B B'], 'maneiras possíveis, dado a hipótese.')
print('Plausibildade da hipótese [A A A B] = ', plausibilidade['A A A B'], 'maneiras possíveis, dado a hipótese.')
print('Plausibildade da hipótese [A A A A] = ', plausibilidade['A A A A'], 'maneiras possíveis, dado a hipótese.')


# ----

# ### Adicionando Prioris
# 

# Imagine que na fábrica que produz bolsas com bolinhas dentro, a informação que um funcionário nos disse é que:
# 
# - Existem poucas bolinhas `Azuis` em cada bolsa, e cada bolsa tem uma chance bem grande de ter uma 1 bolinha Azul. Para nós conseguirmos informar está **intuição**  sobre a quantidade de bolinhas que são mais prováveis que cada bolsa contenha, podemos descrever os pesos mais viáveis. Chamaremos essa nova informação de *à priori*:
#     
# $$ [B B B B]  = 0 $$
# 
# $$ [A B B B]  = 3 $$
# 
# $$ [A A B B]  = 2 $$
# 
# $$ [A A A B]  = 1 $$
# 
# $$ [A A A B]  = 0 $$

# Com essas informações, podemos multiplicar nossas contagens para cada uma das hipóteses.

# In[7]:


# Inserindo a nossa informação a priori no nosso modelo.

priori = {}   # Inicializando um novo dicionário com as prioris.

priori['B B B B'] = 0  
priori['A B B B'] = 3   
priori['A A B B'] = 2  
priori['A A A B'] = 1  
priori['A A A A'] = 0


# In[8]:


# Calculando a posteriori

nova_plausibilidade = {}   # Inicializando um novo dicionário com as posterioris.

nova_plausibilidade['B B B B'] = priori['B B B B'] * plausibilidade['B B B B']
nova_plausibilidade['A B B B'] = priori['A B B B'] * plausibilidade['A B B B']
nova_plausibilidade['A A B B'] = priori['A A B B'] * plausibilidade['A A B B']
nova_plausibilidade['A A A B'] = priori['A A A B'] * plausibilidade['A A A B']
nova_plausibilidade['A A A A'] = priori['A A A A'] * plausibilidade['A A A A']


# Com a nova informação, teremos novas plausabilidade para cada uma das hipóteses sugeridas.

# In[9]:


# Mostrando, para cada uma das hipóteses, o quão plausível são as novas contagens. 

print('Nova plausibildade da hipótese [A B B B] = ', nova_plausibilidade['A B B B'], 'maneiras possíveis, dado a hipótese.')
print('Nova plausibildade da hipótese [A A B B] = ', nova_plausibilidade['A A B B'], 'maneiras possíveis, dado a hipótese.')
print('Nova plausibildade da hipótese [A A A B] = ', nova_plausibilidade['A A A B'], 'maneiras possíveis, dado a hipótese.')
print('Nova plausibildade da hipótese [A A A A] = ', nova_plausibilidade['A A A A'], 'maneiras possíveis, dado a hipótese.')


#  ### Conclusão

# Temos uma intuição lógica muito boa de qual das possíveis bolsas que sugerimos poderia ser. Sabemos que as nossas bolsas hipotéticas que contém apenas bolas azuis $[A A A A]$ ou apenas bolas brancas $[B B B B]$ tem *peso* de $0$ de acontecer, pois sabemos que tem pelo menos $1$ bola Azul e $1$ bola Branca na nossa amostragem.
# 
# Já nossas outras bolsas hipotéticas, tem suas `plausibilidade`positivas de acontecer. As mais plausíveis são as bolsas com maior número de maneiras de acontecer! *Lindo!*

# ----

# O exemplo das *quantidade bolinhas contidas na bolsa* apresenta como é o funcionamento e a construção da lógica de *todos* os modelos bayesianos que iremos construir.
# 
# Em muitos casos, iremos precisar contar infinitas hipóteses. Para isso, iremos usar o computador juntamente com cálculo para conseguir fazer tais contagens.

# Porém existe uma particularidade no desenvolvimento acima: uma contagem poderá ficar muito, mas muito grande, quando o número de bolinhas contidas na bolsa aumenta e também quando aumentamos a quantidade de vezes que retiramos uma bola da urna. Por exemplo, uma urna com 10 bolinhas e 10 retiradas, qual a magnetudo do número de contagens? E 1000 bolinhas? E infinitas? Essa última, apesar de ser parecer absurda, será a que iremos utilizar com mais frequência.
# 
# *Como contar até infinito muitas vezes?*

# Voltando no exemplo das bolinhas, podemos calcular somas relativas, ou seja, qual a proporção de bolas azuis (por exemplo) que existem na amostra?

# In[10]:


# Quantidade total de maneiras possíveis de ocorrer uma das hipóteses

total = sum(nova_plausibilidade.values())
print('A quantidade total: ', total)


# Normalizando a nossa contagem pela total, teremos:

# In[11]:


# Normalizando as contagens plausíveis.

plasusibilidade_normalizada = {}   # Inicializando um novo dicionário.

plasusibilidade_normalizada['B B B B'] = nova_plausibilidade['B B B B'] / total
plasusibilidade_normalizada['A B B B'] = nova_plausibilidade['A B B B'] / total
plasusibilidade_normalizada['A A B B'] = nova_plausibilidade['A A B B'] / total
plasusibilidade_normalizada['A A A B'] = nova_plausibilidade['A A A B'] / total
plasusibilidade_normalizada['A A A A'] = nova_plausibilidade['A A A A'] / total


# In[12]:


print('Nova plausibildade da hipótese [A B B B] = ', round(plasusibilidade_normalizada['B B B B'], 2), 'maneiras possíveis, dado a hipótese.')
print('Nova plausibildade da hipótese [A B B B] = ', round(plasusibilidade_normalizada['A B B B'], 2), 'maneiras possíveis, dado a hipótese.')
print('Nova plausibildade da hipótese [A A B B] = ', round(plasusibilidade_normalizada['A A B B'], 2), 'maneiras possíveis, dado a hipótese.')
print('Nova plausibildade da hipótese [A A A B] = ', round(plasusibilidade_normalizada['A A A B'], 2), 'maneiras possíveis, dado a hipótese.')
print('Nova plausibildade da hipótese [A A A A] = ', round(plasusibilidade_normalizada['A A A A'], 2), 'maneiras possíveis, dado a hipótese.')


# Agora todas as plausibilidades normalizadas estão entre $0$ e $1$, e soma de todos os elementos será, sempre, $1$.

# ----

# Será nossa **probablidade**, isso é, é o **peso** da evidência para cada possibildade.  Essa é a inferência bayesiana! *Lindo!!!*
# 

# A **teoria das probablidade** é o único conjunto de ferramentas que nos permitam trabalhar com números normalizados entre $0$ e $1$.

# *Plausibilidade é a probabilidade*: É um conjunto de números não-negativos que somam $1$. 
# 
# **São os números de maneiras pelas quais cada uma dessas conjecturas poderia ser verdadeira condicionalmente em  todas as evidências!!!**

# A teroria das probabilidades é apenas um conjunto de atalhos para a possibilidades de contagens.
