# 2d-drone-mapping

- autor: Jan Kapsa
- datum: 9.5.2024

### Instalace

Proces instalace spusťte následujícím příkazem:

```
pip install -r requirements.txt
```

### Spouštění skriptu

```
python main.py [-h] [-s {KM,PI}] [-l LIMIT] [-n SKIP_N] dirpath

positional arguments:
  dirpath               cesta k adresáři, který obsahuje snímky

options:
  -h, --help            ukáže tuto zprávu v angličtině
  -s {KM,PI}, --strat {KM,PI}
                        vybere metodu extrakce - zadejte KM pro metodu klíčových bodů a PI pro metodu předchozího obrázku
  -l LIMIT, --limit LIMIT
                        limituje číslo zpracovávaných snímků
  -n SKIP_N, --skip_n SKIP_N
                        přeskočí daný počet prvních snímků v sekvenci
```

### Dataset

Dataset obsahuje 6 různých letů. Více jsou data rozebírána v bakalářské práci. Data nejsou určena pro komerční použití.

### Vygenerování programové dokumentace

Projekt používá Doxyfile pro automatickou tvorbu dokumentace. Vygenerování html stránky spusťte příkazem (za předpokladu, že máte nainstalovaný doxygen):

```
doxygen Doxyfile
```
