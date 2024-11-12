# VFL_demo
## Запуск обучения
Предлагаемый demo-продукт позволяет проводить вертикальное федеративное обучение линейной модели, модели логистической регрессии и софтмакс регрессии (многоклассовой регрессии) совместно несколькими участниками (от двух и более) без необходимости передачи самих данных. При этом обмен служебной информацией, необходимой для обучения модели, защищен криптографически стойким алгоритмом шифрования, что гарантирует конфиденциальность данных и таргетного признака участников обучения относительно друг друга.
Для запуска обучения модели совместно несколькими участниками необходимо:
1) **на стороне каждого участника, не обладающего таргетным признаком (пассивная сторона)**
   - запустить скрипт VFL_server.py, указав в командной строке следующие параметры:
	- -i <id участника, может быть просто порядковым номером участника, значение по умолчанию 1>
	- -p <номер порта компьютера, через который будет осуществляться сетевое соединение во время обучения модели, значение по умолчанию 50055>
	- -t <путь к csv-файлу с обучающим датасетом данного участника, по уполчанию указан файл-пример из папки data данного demo>
	- -v <путь к csv-файлу с тестовым (валидационным) датасетом данного участника, по уполчанию указан файл-пример из папки data данного demo>
	- -d <название столбца данных, содержащего идентификационную информацию записи данных, значение по уполчанию 'ID'>
	- -f <доля случайно выбираемых записей из обучающего и тестового датасетов для выполнения обучения, значение по уполчанию 0.5>
2) **на стороне участника, обладающего таргетным признаком (активная сторона)**
   - указать параметры обучения в конфигурационном файле, пример которого расположен в папке config данного demo:
	- а) заполнить по образцу данные о пассивных сторонах, указав для каждой id, IP-адрес и порт для сетевого взаимодействия, при этом порт должен совпадать с портом, указанным при запуске скрипта VFL_server.py пассивной стороной с тем же id;
	- б) задать параметры обучения: learning_rate для обучаемой модели и число эпох обучения;
	- в) задать значения порога бинаризации для выполнения инференса;
	- г) задать длину в битах ключа криптосхемы Пайе;
	- б) запустить скрипт VFL_client.py, указав в командной строке следующие параметры:
		- -c <путь к конфигурационному файлу, значение по умолчанию 'config/config.conf'>
		- -m <один из доступных типов обучаемой модели из списка 'linear', 'logistic', 'softmax', значение по умолчанию 'softmax'>
		- -t <путь к csv-файлу с обучающим датасетом данного участника, по уполчанию указан файл-пример из папки data данного demo>
		- -v <путь к csv-файлу с тестовым (валидационным) датасетом данного участника, по уполчанию указан файл-пример из папки data данного demo>
		- -d <название столбца данных, содержащего идентификационную информацию записи данных, значение по уполчанию 'ID'>
		- -y <название столбца данных, содержащего признак, значение по уполчанию 'y'>
		- -f <доля случайно выбираемых записей из обучающего и тестового датасетов для выполнения обучения, значение по уполчанию 0.5>

По окончании обучения у каждого из участников обучения будет сохранены параметры общей модели, относящиеся к его данным. 
При проведении инференса параметры модели загружаются каждым участником, после чего выполняется инференс на записях.


## Start training

The provided demo product enables vertical federated learning for linear models, logistic regression models, and softmax (multiclass) regression among multiple participants (from two or more) without needing to share the actual data. The exchange of metadata required for model training is cryptographically secured, ensuring the confidentiality of each participant's data and target labels.

To initiate the federated model training among multiple participants:

1. **On the side of participants without the target label (passive side)**:
   - Run the 'VFL_server.py' script with the following command-line parameters:
     - -i <participant id, which can be a sequential number, default is 1>
     - -p <port number for network connection during training, default is 50055>
     - -t <path to the participant's training dataset in csv format, default is an example file in the demo’s data folder>
     - -v <path to the participant's test (validation) dataset in csv format, default is an example file in the demo’s data folder>
     - -d <column name containing record ID information, default is 'ID'>
     - -f <fraction of randomly selected records from the training and test datasets for model training, default is 0.5>

2. **On the side of the participant with the target label (active side)**:
   - Specify the training parameters in a configuration file, an example of which can be found in the config folder of the demo:
     - a) Fill out information on passive participants, specifying for each their 'id', 'IP address', and port for network communication (the port should match the one specified in 'VFL_server.py' by the passive participant with the same id).
     - b) Define training parameters, such as the model's learning rate and the number of training epochs.
     - c) Set the binarization threshold values for inference.
     - d) Specify the bit length of the Paillier cryptosystem key.
   - Run the 'VFL_client.py' script with the following command-line parameters:
     - -c <path to the configuration file, default is 'config/config.conf'>
     - -m <model type from available options 'linear', 'logistic', 'softmax', default is 'softmax'>
     - -t <path to the training dataset in csv format, default is an example file in the demo’s data folder>
     - -v <path to the test (validation) dataset in csv format, default is an example file in the demo’s data folder>
     - -d <column name containing record ID information, default is 'ID'>
     - -y <column name containing the target label, default is 'y'>
     - -f <fraction of randomly selected records from the training and test datasets, default is 0.5>

After training, each participant will have the model parameters relevant to their own data saved. For inference, the participants load the model parameters and perform inference on their records.
