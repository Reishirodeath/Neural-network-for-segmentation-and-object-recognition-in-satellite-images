import math
import requests
import os

class GoogleMapDownloader: #Инициализация переменных
    def __init__(self, lat, lng, zoom=15):
        self._lat = lat
        self._lng = lng
        self._zoom = zoom

    def getXY(self):   #Пересчёт координат в xy
        tile_size = 256
        numTiles = 1 << self._zoom
        point_x = ((tile_size / 2 + self._lng * tile_size / 360.0)
                                          * numTiles // tile_size)
        sin_y = math.sin(self._lat * (math.pi / 180.0))
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y))
                       * -(tile_size / (2 * math.pi))) * numTiles // tile_size
        return int(point_x), int(point_y)

    def download_tile(self, x, y, save_path):
        # API Goole Maps и проверка на ошибки
        url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={self._zoom}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Тайл сохранён: {save_path}")
        except Exception as e:
            print(f"Ошибка: {e}")

# Пример использования
if __name__ == "__main__":
    # Координаты Екатеринбурга
    lat, lon = 56.824405, 60.608656
    zoom = 15
    downloader = GoogleMapDownloader(lat, lon, zoom)
    x, y = downloader.getXY()

    # Скачивание одиночного тайла
    downloader.download_tile(x, y, "ekaterinburg_tile.png")