# Diplomová práca
Tento repozitár obsahuje diplomovú prácu na tému `Detekcia anomálií vo videozáznamoch z
bezpečnostných kamier pomocou
vision-language modelov` v 
jazyku Python. Práca je vyhotovená na Fakulte elektrotechniky a informatiky v 
Bratislave.

Inštalácia
1. Stiahnite repozitár.
2. Nainštalujte všetky potrebné knižnice príkazom:
pip install -r requirements.txt
3. Na prístup k modelom z Hugging Face je potrebné mať Hugging Face API key.
Zaregistrujte sa na https://huggingface.co/
Vygenerujte si Access Token v nastaveniach účtu
Nastavte token v konzole použitím príkazu:
huggingface-cli login
4. Stiahnite modely Florence-2 a PaliGemma z Hugging Face:
Florence-2: https://huggingface.co/microsoft/Florence-2-large
PaliGemma: https://huggingface.co/google/paligemma-3b-pt-224
5. Spustite Flask aplikáciu v konzole:
python app.py
6. Aplikácia sa otvorí vo webovom prehliadači.

Používanie systému
V prostredí aplikácie možno nahrať obrázok alebo video.
Používateľ si následne zvolí model (napr. Florence- 2 alebo PaliGemma) a typ
promptu.
Po spracovaní vstupu sa zobrazí výsledok vo forme textového výstupu alebo
klasifikácie.
Lokálne modely
Dotrénované modely PaliGemma fine-tuned a Florence-2 fine-tuned nie sú súčas-
ťou repozitára z dôvodu ich veľkosti. Aplikácia je spustiteľná aj bez týchto modelov,
avšak pri ich výbere sa nevykoná žiadna klasifikácia. Použiť sa dajú len predtrénované
modely.
