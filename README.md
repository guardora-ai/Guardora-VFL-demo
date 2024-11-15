# Guardora-VFL-demo

Предлагаемый demo-продукт позволяет проводить вертикальное федеративное обучение линейной модели, модели логистической регрессии и софтмакс (мультиклассовой) регрессии совместно несколькими участниками (от двух и более) без необходимости передачи самих данных. При этом обмен служебной информацией, необходимой для обучения модели, защищен криптографически стойким алгоритмом шифрования, что гарантирует конфиденциальность данных и таргетного признака участников обучения относительно друг друга. 

## Использоваиние SSL\TLS
По умолчанию коммуникационные каналы участников обучения защищаются SSL\TLS-шифрованием на основе сертификатов. Данную функцию можно отключить, указав соответствующий параметр командной строки при запуске обучения. Для реализации шифрования требуются сертификаты, примеры которых находятся в папке cert, их достаточно для проведения экспериментального обучения на localhost, однако возможно использование сертификатов любых других сторон обучения, для чего достаточно указать пути к соответствующим файлам в командной строке запуска. При отсутствии готовых сертификатов других участников имеется возможность сгенерировать необходимый набор самоподписанных сертификатов и ключей с использованием скрипта 'cert/generate_certificate.sh', который запускается командой

`chmod a+x generate_certificate.sh;bash generate_certificate.sh <адрес_участника>`, 

где адрес участника может представлять собой IP-адрес или доменное имя, после чего в командной строке запуска обучения необходимо будет указать пути к сформированным файлам сертификатов и ключу.

## Запуск обучения
Для запуска обучения модели совместно несколькими участниками необходимо:
1) **на стороне каждого участника, не обладающего таргетным признаком (пассивная сторона)**:
   - запустить скрипт VFL_server.py, указав в командной строке следующие параметры:
     - -i <id участника, может быть просто порядковым номером участника, значение по умолчанию 1>
     - -p <номер порта компьютера, через который будет осуществляться сетевое соединение во время обучения модели, значение по умолчанию 50055>
     - -t <путь к csv-файлу с обучающим датасетом данного участника, по уполчанию указан файл-пример из папки data данного demo>
     - -v <путь к csv-файлу с тестовым (валидационным) датасетом данного участника, по уполчанию указан файл-пример из папки data данного demo>
     - -d <название столбца данных, содержащего идентификационную информацию записи данных, значение по уполчанию 'ID'>
     - -f <доля случайно выбираемых записей из обучающего и тестового датасетов для выполнения обучения, значение по уполчанию 0.5>
     - -ns <флаг no_secure, его упоминание отключает SSL\TLS-шифрование каналов связи участников, по умолчанию SSL\TLS-шифрование каналов включено>
     - -cp <путь к crt-файлу сертификата сервера для выполнения SSL\TLS-шифрование каналов связи участников, по уполчанию указан localhost-сертификат cert/127.0.0.1.crt>
     - -kp <путь к key-файлу ключа сервера для выполнения SSL\TLS-шифрование каналов связи участников, по уполчанию указан localhost-ключ cert/127.0.0.1.key>
2) **на стороне участника, обладающего таргетным признаком (активная сторона)**:
   - указать параметры обучения в конфигурационном файле, пример которого расположен в папке config данного demo:
     - заполнить по образцу данные о пассивных сторонах, указав для каждой id, IP-адрес (доменное имя) и порт для сетевого взаимодействия, при этом порт должен совпадать с портом, указанным при запуске скрипта VFL_server.py пассивной стороной с тем же id, а IP-адрес (доменное имя) должен соответствовать сертификату в случае использования SSL\TLS-шифрования каналов связи;
     - задать параметры обучения: learning_rate для обучаемой модели и число эпох обучения;
     - задать значения порога бинаризации для выполнения инференса;
     - задать длину в битах ключа криптосхемы Пайе;
   - запустить скрипт VFL_client.py, указав в командной строке следующие параметры:
     - -c <путь к конфигурационному файлу, значение по умолчанию 'config/config.conf'>
     - -m <один из доступных типов обучаемой модели из списка 'linear', 'logistic', 'softmax', значение по умолчанию 'softmax'>
     - -t <путь к csv-файлу с обучающим датасетом данного участника, по уполчанию указан файл-пример из папки data данного demo>
     - -v <путь к csv-файлу с тестовым (валидационным) датасетом данного участника, по уполчанию указан файл-пример из папки data данного demo>
     - -d <название столбца данных, содержащего идентификационную информацию записи данных, значение по уполчанию 'ID'>
     - -y <название столбца данных, содержащего признак, значение по уполчанию 'y'>
     - -f <доля случайно выбираемых записей из обучающего и тестового датасетов для выполнения обучения, значение по уполчанию 0.5>
     - -ns <флаг no_secure, его упоминание отключает SSL\TLS-шифрование каналов связи участников, по умолчанию SSL\TLS-шифрование каналов включено>
     - -cp <путь к crt-файлу сертификата сервера для выполнения SSL\TLS-шифрование каналов связи участников, по уполчанию указан localhost-сертификат cert/127.0.0.1.crt>
     - -kp <путь к key-файлу ключа сервера для выполнения SSL\TLS-шифрование каналов связи участников, по уполчанию указан localhost-ключ cert/127.0.0.1.key>
     - -rp <путь к корневому сертификату для выполнения SSL\TLS-шифрование каналов связи участников, по уполчанию указан корневой localhost-сертификат cert/rootCA.crt>

По окончании обучения у каждого из участников обучения будет сохранены параметры общей модели, относящиеся к его данным. 
При проведении инференса параметры модели загружаются каждым участником, после чего выполняется инференс на записях.


# **Guardora-VFL-Demo**

The proposed demo product enables **vertical federated learning** for linear models, logistic regression, and softmax (multiclass) regression, allowing multiple participants (two or more) to collaborate without sharing raw data. Metadata exchanges necessary for model training are secured by a cryptographically robust encryption algorithm, ensuring the confidentiality of each participant's data and target labels.

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
- `-t` `<Path to the participant's training dataset (CSV format); default: example file in the `data` folder>`
- `-v` `<Path to the participant's test/validation dataset (CSV format); default: example file in the `data` folder>`
- `-d` `<Name of the column containing record IDs; default: 'ID'>`
- `-f` `<Fraction of randomly selected records from the training and test datasets for model training; default: 0.5>`
- `-ns` `<Flag to disable SSL/TLS encryption (optional); by default, SSL/TLS encryption is enabled>`
- `-cp` `<Path to the server certificate file for SSL/TLS encryption; default: `cert/127.0.0.1.crt>`>
- `-kp` `<Path to the server key file for SSL/TLS encryption; default: `cert/127.0.0.1.key>`>

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
     - `-t` `<Path to the training dataset (CSV format); default: example file in `data` folder>`
     - `-v` `<Path to the test/validation dataset (CSV format); default: example file in `data` folder>`
     - `-d` `<Name of the column containing record IDs; default: 'ID'>`
     - `-y` `<Name of the column containing the target labels; default: 'y'>`
     - `-f` `<Fraction of randomly selected records for training/testing; default: 0.5>`
     - `-ns` `<Flag to disable SSL/TLS encryption (optional); by default, SSL/TLS encryption is enabled>`
     - `-cp` `<Path to server certificate file for SSL/TLS encryption; default: `cert/127.0.0.1.crt>`>
     - `-kp` `<Path to server key file for SSL/TLS encryption; default: `cert/127.0.0.1.key>`>
     - `-rp` `<Path to the root certificate for SSL/TLS encryption; default: `cert/rootCA.crt>`>

---

### **Post-Training and Inference**
After training, each participant retains the parameters of the global model relevant to their data. During inference, each participant loads their respective model parameters and performs inference on their records.