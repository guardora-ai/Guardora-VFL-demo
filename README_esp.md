# **Guardora-VFL-Demo**

El producto demo propuesto permite el **aprendizaje federado vertical** para modelos lineales, regresión logística y regresión softmax (multiclase), permitiendo que múltiples participantes (dos o más) colaboren sin compartir datos en bruto. Los intercambios de metadatos necesarios para el entrenamiento del modelo están protegidos por un algoritmo de cifrado criptográficamente robusto, asegurando la confidencialidad de los datos y las etiquetas objetivo de cada participante.

---

### **Instrucciones de Instalación**

#### **Requisitos Previos**
1. Asegúrate de tener Python 3.9 o superior instalado en tu ordenador:
   ```bash
   python --version
   ```

2. Verifica que el gestor de paquetes `pip` esté instalado:
   ```bash
   pip --version
   ```

#### **Descarga del Proyecto e Instalación de Dependencias**
1. Clona el repositorio desde GitHub:
   ```bash
   git clone https://github.com/guardora-ai/Guardora-VFL-demo.git
   ```
   > Si no tienes acceso al repositorio, por favor envía una solicitud al canal `#federated-learning` en la Comunidad Guardora en Discord o contacta a `iam@guardora.ai`.

2. Ve al directorio del proyecto:
   ```bash
   cd Guardora-VFL-demo/
   ```

3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Uso de SSL/TLS**
Por defecto, los canales de comunicación entre los participantes están protegidos mediante cifrado SSL/TLS basado en certificados. Esta función puede desactivarse especificando el parámetro de línea de comandos correspondiente al iniciar el entrenamiento. Para el cifrado, se requieren certificados. Se proporcionan ejemplos en la carpeta `cert`, adecuados para entrenamientos experimentales en `localhost`. También se pueden usar certificados personalizados para otros participantes especificando sus rutas en la línea de comandos. Si no hay certificados disponibles, se pueden generar certificados y claves autofirmados utilizando el script `cert/generate_certificate.sh`, ejecutado con el comando:

```bash
chmod a+x generate_certificate.sh; bash generate_certificate.sh <participant_address>
```

Aquí, `<participant_address>` representa una dirección IP o un nombre de dominio. Las rutas de los archivos generados de certificado y clave deben proporcionarse luego en el comando de inicio del entrenamiento.

---

### **Proceso de Entrenamiento**

#### **Participantes Pasivos (Sin Etiquetas Objetivo)**:
Cada participante sin etiquetas objetivo debe ejecutar el script `VFL_server.py` con los siguientes parámetros:

- `-i` `<ID del participante (por ejemplo, número secuencial); por defecto: 1>`
- `-p` `<Número de puerto para conexiones de red durante el entrenamiento; por defecto: 50055>`
- `-t` `<Ruta al conjunto de datos de entrenamiento del participante (formato CSV); por defecto: archivo de entrenamiento en la carpeta 'data'>`
- `-v` `<Ruta al conjunto de datos de prueba/validación del participante (formato CSV); por defecto: archivo de prueba en la carpeta 'data'>`
- `-d` `<Nombre de la columna que contiene los IDs de los registros; por defecto: 'ID'>`
- `-f` `<Fracción de registros seleccionados aleatoriamente de los conjuntos de entrenamiento y prueba para el entrenamiento del modelo; por defecto: 0.5>`
- `-ns` `<Indicador para desactivar el cifrado SSL/TLS (opcional); por defecto, el cifrado SSL/TLS está habilitado>`
- `-cp` `<Ruta al certificado del servidor para el cifrado SSL/TLS; por defecto: 'cert/127.0.0.1.crt'>`
- `-kp` `<Ruta a la clave del servidor para el cifrado SSL/TLS; por defecto: 'cert/127.0.0.1.key'>`

#### **Participante Activo (Con Etiquetas Objetivo)**:
El participante que posea las etiquetas objetivo debe configurar los parámetros de entrenamiento en un archivo de configuración (se proporciona un ejemplo en la carpeta `config`). Los pasos incluyen:

1. **Definir Participantes Pasivos**:
   - Especifica el `ID`, la `dirección IP` (o nombre de dominio) y el puerto de cada participante pasivo. Asegúrate de que el puerto coincida con el usado al ejecutar `VFL_server.py` para el participante correspondiente, y que la `dirección IP`/dominio coincida con el certificado (si SSL/TLS está habilitado).

2. **Establecer Parámetros de Entrenamiento**:
   - Especifica la `tasa de aprendizaje` y el `número de épocas`.

3. **Establecer Umbral para Clasificación Binaria**:
   - Define los valores de umbral para la inferencia.

4. **Establecer Longitud de la Clave de Cifrado**:
   - Especifica la longitud de la clave (en bits) para el **sistema criptográfico de Paillier**.

5. **Ejecutar el Script de Entrenamiento**:
   - Ejecuta el script `VFL_client.py` con los siguientes parámetros:
     - `-c` `<Ruta al archivo de configuración; por defecto: 'config/config.conf'>`
     - `-m` `<Tipo de modelo: 'linear', 'logistic' o 'softmax'; por defecto: 'softmax'>`
     - `-t` `<Ruta al conjunto de datos de entrenamiento (formato CSV); por defecto: archivo de entrenamiento en la carpeta 'data'>`
     - `-v` `<Ruta al conjunto de datos de prueba/validación (formato CSV); por defecto: archivo de prueba en la carpeta 'data'>`
     - `-d` `<Nombre de la columna que contiene los IDs de los registros; por defecto: 'ID'>`
     - `-y` `<Nombre de la columna que contiene las etiquetas objetivo; por defecto: 'y'>`
     - `-f` `<Fracción de registros seleccionados aleatoriamente para entrenamiento/prueba; por defecto: 0.5>`
     - `-ns` `<Indicador para desactivar el cifrado SSL/TLS (opcional); por defecto, el cifrado SSL/TLS está habilitado>`
     - `-cp` `<Ruta al certificado del servidor para el cifrado SSL/TLS; por defecto: 'cert/127.0.0.1.crt'>`
     - `-kp` `<Ruta a la clave del servidor para el cifrado SSL/TLS; por defecto: 'cert/127.0.0.1.key'>`
     - `-rp` `<Ruta al certificado raíz para el cifrado SSL/TLS; por defecto: 'cert/rootCA.crt'>`

---

### **Inferencia**
Después del entrenamiento, cada participante conserva los parámetros del modelo global relevantes para sus datos. Durante la inferencia, cada participante carga sus respectivos parámetros del modelo y realiza inferencia sobre sus registros.