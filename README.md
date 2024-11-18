# **Guardora-VFL-Demo**

Предлагаемый demo-продукт позволяет проводить **вертикальное федеративное обучение** линейной модели, модели логистической регрессии и софтмакс (мультиклассовой) регрессии совместно несколькими участниками (от двух и более) без необходимости передачи самих данных. При этом обмен служебной информацией, необходимой для обучения модели, защищен криптографически стойким алгоритмом шифрования, что гарантирует конфиденциальность данных и таргетного признака участников обучения относительно друг друга.

---

### **Инструкции по установке**

#### **Предварительные требования**
1. Убедитесь, что на вашем компьютере установлена версия Python 3.9 или выше:
   ```bash
   python --version
   ```

2. Проверьте, что установлен менеджер пакетов `pip`:
   ```bash
   pip --version
   ```

#### **Скачивание проекта и установка зависимостей**
1. Клонируйте репозиторий с GitHub:
   ```bash
   git clone https://github.com/guardora-ai/Guardora-VFL-demo.git
   ```
   > Если у вас нет доступа к репозиторию, отправьте запрос в канал `#federated-learning` в сообществе Guardora на Discord или напишите на почту `guardora@guardora.ru`.

2. Перейдите в каталог проекта:
   ```bash
   cd Guardora-VFL-demo/
   ```

3. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Использование SSL/TLS**

По умолчанию коммуникационные каналы участников обучения защищаются SSL/TLS-шифрованием на основе сертификатов. Данную функцию можно отключить, указав соответствующий параметр командной строки при запуске обучения. Для реализации шифрования требуются сертификаты, примеры которых находятся в папке `cert`. Их достаточно для проведения экспериментального обучения на `localhost`. Однако возможно использование сертификатов любых других сторон обучения, для чего достаточно указать пути к соответствующим файлам в командной строке запуска.

Если готовые сертификаты отсутствуют, можно сгенерировать необходимый набор самоподписанных сертификатов и ключей с использованием скрипта `cert/generate_certificate.sh`. Для выполнения скрипта используйте команду:

```bash
chmod a+x generate_certificate.sh; bash generate_certificate.sh <адрес_участника>
```

Здесь `<адрес_участника>` может быть IP-адресом или доменным именем. После генерации необходимо указать пути к сформированным сертификатам и ключам в командной строке запуска обучения.

---

### **Запуск обучения**

#### **Пассивные участники (без таргетного признака)**

Каждый участник без таргетного признака должен запустить скрипт `VFL_server.py` с использованием следующих параметров командной строки:

- `-i` `<ID участника, например, порядковый номер; значение по умолчанию: 1>`  
- `-p` `<Номер порта для сетевых соединений во время обучения; значение по умолчанию: 50055>`  
- `-t` `<Путь к обучающему датасету (CSV-файл); значение по умолчанию: train файл из папки 'data'>`  
- `-v` `<Путь к тестовому/валидационному датасету (CSV-файл); значение по умолчанию: test файл из папки 'data'>`  
- `-d` `<Название столбца, содержащего идентификаторы записей; значение по умолчанию: 'ID'>`  
- `-f` `<Доля случайно выбираемых записей из обучающих и тестовых данных для выполнения обучения; значение по умолчанию: 0.5>`  
- `-ns` `<Флаг отключения SSL/TLS-шифрования (опционально); по умолчанию SSL/TLS включено>`  
- `-cp` `<Путь к серверному сертификату для SSL/TLS; значение по умолчанию: 'cert/127.0.0.1.crt'>`  
- `-kp` `<Путь к серверному ключу для SSL/TLS; значение по умолчанию: 'cert/127.0.0.1.key'>`  

#### **Активный участник (с таргетным признаком)**

Участник, обладающий таргетным признаком, должен указать необходимые параметры в конфигурационном файле (пример файла можно найти в папке `config`) и запустить обучение. Т.е. выполнить следующие шаги:

1. **Указать пассивных участников**:
   - Определить `id`, `IP-адрес` (или доменное имя) и порт для каждого пассивного участника. Порт должен совпадать с указанным в `VFL_server.py`, а `IP-адрес`/доменное имя — соответствовать сертификату (если SSL/TLS включено).

2. **Задать параметры обучения**:
   - Указать `learning_rate` и число эпох.

3. **Установить порог бинаризации**:
   - Задать значения порога бинаризации для выполнения инференса.

4. **Указать длину ключа шифрования**:
   - Определить длину ключа (в битах) для **криптосистемы Пайе**.

5. **Запустить скрипт обучения**:
   - Выполнить скрипт `VFL_client.py` с использованием следующих параметров:

     - `-c` `<Путь к конфигурационному файлу; значение по умолчанию: 'config/config.conf'>`  
     - `-m` `<Тип модели: 'linear', 'logistic' или 'softmax'; значение по умолчанию: 'softmax'>`  
     - `-t` `<Путь к обучающему датасету (CSV-файл); значение по умолчанию: train файл из папки 'data'>`  
     - `-v` `<Путь к тестовому/валидационному датасету (CSV-файл); значение по умолчанию: test файл из папки 'data'>`  
     - `-d` `<Название столбца, содержащего идентификаторы записей; значение по умолчанию: 'ID'>`  
     - `-y` `<Название столбца, содержащего таргетный признак; значение по умолчанию: 'y'>`  
     - `-f` `<Доля случайно выбираемых записей для обучения/тестирования; значение по умолчанию: 0.5>`  
     - `-ns` `<Флаг отключения SSL/TLS-шифрования (опционально); по умолчанию SSL/TLS включено>`  
     - `-cp` `<Путь к серверному сертификату для SSL/TLS; значение по умолчанию: 'cert/127.0.0.1.crt'>`  
     - `-kp` `<Путь к серверному ключу для SSL/TLS; значение по умолчанию: 'cert/127.0.0.1.key'>`  
     - `-rp` `<Путь к корневому сертификату для SSL/TLS; значение по умолчанию: 'cert/rootCA.crt'>`  

---

### **Инференс**

После завершения обучения у каждого участника будут сохранены параметры общей модели, относящиеся к его данным. При проведении инференса параметры модели загружаются каждым участником, после чего выполняется инференс на записях.

---


# **Guardora-VFL-Demo**

The proposed demo product enables **vertical federated learning** for linear models, logistic regression, and softmax (multiclass) regression, allowing multiple participants (two or more) to collaborate without sharing raw data. Metadata exchanges necessary for model training are secured by a cryptographically robust encryption algorithm, ensuring the confidentiality of each participant's data and target labels.

---

### **Installation Instructions**

#### **Prerequisites**
1. Ensure Python 3.9 or higher is installed on your computer:
   ```bash
   python --version
   ```

2. Verify that the `pip` package manager is installed:
   ```bash
   pip --version
   ```

#### **Downloading the Project and Installing Dependencies**
1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/guardora-ai/Guardora-VFL-demo.git
   ```
   > If you do not have access to the repository, please send a request to the `#federated-learning` channel in the Guardora Community on Discord or contact `iam@guardora.ai`.

2. Navigate to the project directory:
   ```bash
   cd Guardora-VFL-demo/
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### **SSL/TLS Usage**
By default, communication channels between participants are protected using SSL/TLS encryption based on certificates. This feature can be disabled by specifying the appropriate command-line parameter during training initialization. For encryption, certificates are required. Examples are provided in the `cert` folder, suitable for experimental training on `localhost`. Custom certificates for other participants can also be used by specifying their paths in the command line. If no ready-made certificates are available, self-signed certificates and keys can be generated using the `cert/generate_certificate.sh` script, executed with the command:

```bash
chmod a+x generate_certificate.sh; bash generate_certificate.sh <participant_address>
```

Here, `<participant_address>` represents an IP address or domain name. Generated certificate and key file paths must then be provided in the training initialization command.

---

### **Training Process**

#### **Passive Participants (No Target Labels)**:
Each participant without the target label should execute the `VFL_server.py` script with the following parameters:

- `-i` `<Participant ID (e.g., sequential number); default: 1>`
- `-p` `<Port number for network connections during training; default: 50055>`
- `-t` `<Path to the participant's training dataset (CSV format); default: train file in the 'data' folder>`
- `-v` `<Path to the participant's test/validation dataset (CSV format); default: test file in the 'data' folder>`
- `-d` `<Name of the column containing record IDs; default: 'ID'>`
- `-f` `<Fraction of randomly selected records from the training and test datasets for model training; default: 0.5>`
- `-ns` `<Flag to disable SSL/TLS encryption (optional); by default, SSL/TLS encryption is enabled>`
- `-cp` `<Path to the server certificate for SSL/TLS encryption; default: 'cert/127.0.0.1.crt'>`
- `-kp` `<Path to the server key for SSL/TLS encryption; default: 'cert/127.0.0.1.key'>`

#### **Active Participant (With Target Labels)**:
The participant possessing the target labels must configure the training parameters in a configuration file (example provided in the `config` folder). Steps include:

1. **Define Passive Participants**:
   - Specify `id`, `IP address` (or domain name), and port for each passive participant. Ensure the port matches the one used when running `VFL_server.py` for the corresponding participant, and the `IP address`/domain matches the certificate (if SSL/TLS is enabled).

2. **Set Training Parameters**:
   - Specify the `learning_rate` and `number of epochs`.

3. **Set Binary Classification Threshold**:
   - Define the threshold values for inference.

4. **Set Encryption Key Length**:
   - Specify the key length (in bits) for the **Paillier cryptosystem**.

5. **Run Training Script**:
   - Execute the `VFL_client.py` script with the following parameters:
     - `-c` `<Path to the configuration file; default: 'config/config.conf'>`
     - `-m` `<Type of model: 'linear', 'logistic', or 'softmax'; default: 'softmax'>`
     - `-t` `<Path to the training dataset (CSV format); default: train file in 'data' folder>`
     - `-v` `<Path to the test/validation dataset (CSV format); default: test file in 'data' folder>`
     - `-d` `<Name of the column containing record IDs; default: 'ID'>`
     - `-y` `<Name of the column containing the target labels; default: 'y'>`
     - `-f` `<Fraction of randomly selected records for training/testing; default: 0.5>`
     - `-ns` `<Flag to disable SSL/TLS encryption (optional); by default, SSL/TLS encryption is enabled>`
     - `-cp` `<Path to the server certificate for SSL/TLS encryption; default: 'cert/127.0.0.1.crt'>`
     - `-kp` `<Path to the server key for SSL/TLS encryption; default: 'cert/127.0.0.1.key'>`
     - `-rp` `<Path to the root certificate for SSL/TLS encryption; default: 'cert/rootCA.crt'>`

---

### **Inference**
After training, each participant retains the parameters of the global model relevant to their data. During inference, each participant loads their respective model parameters and performs inference on their records.