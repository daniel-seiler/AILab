# Project

A classificator to detect healthy vs diseased leaves

<cite>xiangm. Computer Vision - XM. https://kaggle.com/competitions/computer-vision-xm, 2024. Kaggle.</cite>
<cite>xiangm. Computer Vision - XM. https://kaggle.com/competitions/computer-vision-xm, 2024. Kaggle.</cite>

## Gettings started

1. Add Kaggle API-Key
    1. Go to your [Kaggle Account](https://www.kaggle.com/settings)
    2. Under API click *Create New Token*
    3. Add the downloaded file under `~/.kaggle/kaggle.json` on Linux, OSX, and other UNIX-based operating systems, and at `C:\Users\<Windows-username>\.kaggle\kaggle.json` on Windows

2. Run notebook

## Server for the fun of it

To upload your own images or use the camera on your phone you can start the webserver with this command from the project directory:

```bash
python -m uvicorn server:app
```

It uses the latest model inside this path. Open http://localhost:8080 to upload your own images and get a result.