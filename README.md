# Bus Sensor Project

Questo progetto implementa un sistema di conteggio persone in tempo reale utilizzando modelli di intelligenza artificiale, telecamere e integrazione con InfluxDB per il monitoraggio dei dati.

---

## Funzionalità principali

1. **Conteggio Persone**: Utilizza YOLO per il rilevamento degli oggetti e un modello di predizione TFLite per stimare il numero di persone presenti.
2. **Predizione Delta**: Calcola la differenza nel conteggio rispetto al frame precedente e regola il tempo di attesa per la prossima inferenza.
3. **Integrazione con InfluxDB**: Invia i dati raccolti (conteggio, delta, predizione) a un server InfluxDB per l'archiviazione e l'analisi.
4. **Sistema Watchdog**: Garantisce l'affidabilità del sistema, riavviandolo in caso di errori prolungati.
5. **Modalità Display**: Facoltativa per la visualizzazione in tempo reale del video con annotazioni.
6. **Log degli errori**: Scrive errori e problemi operativi in un file di log.

---

## Prerequisiti

- **Hardware**:
  - Raspberry Pi o hardware equivalente con supporto per Python e telecamere.
  - Telecamera IP o altra sorgente video compatibile con OpenCV.

- **Software**:
  - Python 3.8 o superiore.
  - Librerie:
    - `opencv-python`
    - `numpy`
    - `requests`
    - `tflite_runtime`
    - `ultralytics`
    - `pywatchdog`
    - `influxdb-client`

---

## Installazione

1. Clona questo repository:
   ```bash
   git clone https://github.com/username/bus-sensor.git
   cd bus-sensor
   ```

2. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

3. Configura il file `config.json` con i tuoi parametri:
   ```json
   {
       "influx": {
           "url": "http://tuo_server_influxdb:8086",
           "org": "organizzazione",
           "token": "tuo_token",
           "bucket": "nome_bucket",
           "host": "host",
           "location": "localita",
           "room": "stanza",
           "urlAdriabus": "http://url_adriabus_api"
       },
       "application": {
           "log-file-path": "path/log/errors.log",
           "source": "rtsp://tuo_stream_rtsp",
           "consumer-max-downtime": 900,
           "model-path": "path/modello/yolov9c.pt",
           "inference-conf": 0.5,
           "min-wait": 60,
           "max-wait": 600,
           "frame-to-skip": 15,
           "save-last-frame": true,
           "last-frame-path": "path/ultimo_frame.jpg"
       }
   }
   ```

4. Avvia l'applicazione:
   ```bash
   python main.py [--display]
   ```

   - Usa `--display` per visualizzare i risultati in tempo reale.

---

## Struttura del progetto

- `main.py`: Script principale che gestisce il conteggio, la predizione e l'invio dei dati.
- `utils.py`: Funzioni di utilità, inclusa la gestione di InfluxDB.
- `config.json`: File di configurazione personalizzabile.

---

## Configurazione Watchdog

Il Watchdog monitora il sistema per garantire che non ci siano blocchi o timeout prolungati. Imposta un timeout di 300 secondi e, in caso di errori, riavvia automaticamente l'applicazione.

---

## API Adriabus

I dati del conteggio sono inviati all'API Adriabus tramite una richiesta POST. Struttura della richiesta:

```json
{
    "CodiceLocalita": "Codice della località",
    "NumPersone": "Numero di persone rilevate",
    "NumPersonePrediction": "Predizione del numero di persone",
    "DataOraEvento": "Timestamp",
    "Note": "Note aggiuntive"
}
```

---

## Log degli errori

Gli errori operativi sono registrati nel file specificato in `log-file-path` nel file di configurazione. In caso di errori critici, il sistema si riavvia automaticamente dopo 1800 secondi.

---
