# Lipik-AI-ispit
Detekcija lopte i aktivacija odgovarajuće kamere 🏀

Jednostavan projekt za automatski odabir kamere (lijeva/desna strana terena).

Ideja je detektirati loptu i odlučiti koja kamera treba biti “uključena”.

## Kako radi

* YOLOv8 model detektira košarkašku loptu na oba videa
* Zadržava se samo jedna detekcija po frameu
* Jednostavna logika praćenja (najbliže prethodnoj poziciji)
* Koristi se buffer (~10 frameova) za pogled "u budućnost"
* Kamera se mijenja ako druga kamera češće vidi loptu u narednim frameovima

## Izlaz

* JSON datoteka s vremenima promjene kamere

## Pokretanje

```bash
pip install ultralytics opencv-python
python main.py
```
