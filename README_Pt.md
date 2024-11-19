# **Guardora-VFL-Demo**

O produto de demonstração proposto permite o **aprendizado federado vertical** para modelos lineares, regressão logística e regressão softmax (multiclasse), permitindo que múltiplos participantes (dois ou mais) colaborem sem compartilhar dados brutos. As trocas de metadados necessárias para o treinamento do modelo são protegidas por um algoritmo de criptografia robusto, garantindo a confidencialidade dos dados e das etiquetas-alvo de cada participante.

---

### **Instruções de Instalação**

#### **Pré-requisitos**
1. Certifique-se de que o Python 3.9 ou superior está instalado no seu computador:
   ```bash
   python --version
   ```

2. Verifique se o gerenciador de pacotes `pip` está instalado:
   ```bash
   pip --version
   ```

#### **Baixando o Projeto e Instalando Dependências**
1. Clone o repositório do GitHub:
   ```bash
   git clone https://github.com/guardora-ai/Guardora-VFL-demo.git
   ```
   > Caso você não tenha acesso ao repositório, envie uma solicitação para o canal `#federated-learning` na Comunidade Guardora no Discord ou entre em contato com `iam@guardora.ai`.

2. Navegue até o diretório do projeto:
   ```bash
   cd Guardora-VFL-demo/
   ```

3. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Uso de SSL/TLS**
Por padrão, os canais de comunicação entre os participantes são protegidos usando criptografia SSL/TLS baseada em certificados. Esta funcionalidade pode ser desativada especificando o parâmetro apropriado na linha de comando durante a inicialização do treinamento. Para criptografia, são necessários certificados. Exemplos estão disponíveis na pasta `cert`, adequados para treinamentos experimentais em `localhost`. Certificados personalizados para outros participantes também podem ser usados especificando seus caminhos na linha de comando. Caso não haja certificados prontos, certificados e chaves autofirmados podem ser gerados usando o script `cert/generate_certificate.sh`, executado com o comando:

```bash
chmod a+x generate_certificate.sh; bash generate_certificate.sh <participant_address>
```

Aqui, `<participant_address>` representa um endereço IP ou nome de domínio. Os caminhos dos arquivos de certificado e chave gerados devem ser fornecidos no comando de inicialização do treinamento.

---

### **Processo de Treinamento**

#### **Participantes Passivos (Sem Etiquetas-Alvo)**:
Cada participante sem etiquetas-alvo deve executar o script `VFL_server.py` com os seguintes parâmetros:

- `-i` `<ID do participante (ex.: número sequencial); padrão: 1>`
- `-p` `<Número da porta para conexões de rede durante o treinamento; padrão: 50055>`
- `-t` `<Caminho para o conjunto de dados de treinamento do participante (formato CSV); padrão: arquivo de treinamento na pasta 'data'>`
- `-v` `<Caminho para o conjunto de dados de teste/validação do participante (formato CSV); padrão: arquivo de teste na pasta 'data'>`
- `-d` `<Nome da coluna que contém os IDs dos registros; padrão: 'ID'>`
- `-f` `<Fração de registros selecionados aleatoriamente dos conjuntos de treinamento e teste para o treinamento do modelo; padrão: 0.5>`
- `-ns` `<Flag para desativar a criptografia SSL/TLS (opcional); por padrão, a criptografia SSL/TLS está ativada>`
- `-cp` `<Caminho para o certificado do servidor para criptografia SSL/TLS; padrão: 'cert/127.0.0.1.crt'>`
- `-kp` `<Caminho para a chave do servidor para criptografia SSL/TLS; padrão: 'cert/127.0.0.1.key'>`

#### **Participante Ativo (Com Etiquetas-Alvo)**:
O participante que possui as etiquetas-alvo deve configurar os parâmetros de treinamento em um arquivo de configuração (exemplo disponível na pasta `config`). Os passos incluem:

1. **Definir Participantes Passivos**:
   - Especifique `id`, `endereço IP` (ou nome de domínio) e porta para cada participante passivo. Certifique-se de que a porta coincide com a usada ao executar `VFL_server.py` para o participante correspondente, e que o `endereço IP`/domínio coincide com o certificado (se SSL/TLS estiver ativado).

2. **Definir Parâmetros de Treinamento**:
   - Especifique a `taxa de aprendizado` e o `número de épocas`.

3. **Definir Limite para Classificação Binária**:
   - Defina os valores de limite para inferência.

4. **Definir o Comprimento da Chave de Criptografia**:
   - Especifique o comprimento da chave (em bits) para o **sistema criptográfico de Paillier**.

5. **Executar o Script de Treinamento**:
   - Execute o script `VFL_client.py` com os seguintes parâmetros:
     - `-c` `<Caminho para o arquivo de configuração; padrão: 'config/config.conf'>`
     - `-m` `<Tipo de modelo: 'linear', 'logistic' ou 'softmax'; padrão: 'softmax'>`
     - `-t` `<Caminho para o conjunto de dados de treinamento (formato CSV); padrão: arquivo de treinamento na pasta 'data'>`
     - `-v` `<Caminho para o conjunto de dados de teste/validação (formato CSV); padrão: arquivo de teste na pasta 'data'>`
     - `-d` `<Nome da coluna que contém os IDs dos registros; padrão: 'ID'>`
     - `-y` `<Nome da coluna que contém as etiquetas-alvo; padrão: 'y'>`
     - `-f` `<Fração de registros selecionados aleatoriamente para treinamento/teste; padrão: 0.5>`
     - `-ns` `<Flag para desativar a criptografia SSL/TLS (opcional); por padrão, a criptografia SSL/TLS está ativada>`
     - `-cp` `<Caminho para o certificado do servidor para criptografia SSL/TLS; padrão: 'cert/127.0.0.1.crt'>`
     - `-kp` `<Caminho para a chave do servidor para criptografia SSL/TLS; padrão: 'cert/127.0.0.1.key'>`
     - `-rp` `<Caminho para o certificado raiz para criptografia SSL/TLS; padrão: 'cert/rootCA.crt'>`

---

### **Inferência**
Após o treinamento, cada participante retém os parâmetros do modelo global relevantes para seus dados. Durante a inferência, cada participante carrega seus respectivos parâmetros do modelo e realiza inferência sobre seus registros.