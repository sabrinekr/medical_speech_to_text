"""Generate German medical audio samples for testing."""

from gtts import gTTS
from pathlib import Path

# Create test_audio directory
output_dir = Path("test_audio")
output_dir.mkdir(exist_ok=True)

# Medical dictation scenarios
scenarios = [
    {
        "filename": "01_bronchitis.mp3",
        "text": """
        Patientin Müller, Maria, vierundvierzig Jahre alt.
        Hauptbeschwerde: Seit drei Tagen anhaltender Husten mit gelblich-grünem Auswurf.
        Die Patientin berichtet über Fieber bis 38,5 Grad Celsius und Abgeschlagenheit.

        Befunde: Auskultation der Lunge ergibt feuchte Rasselgeräusche beidseits basal.
        Temperatur aktuell 37,8 Grad. Herzfrequenz 88 Schläge pro Minute.
        Keine Dyspnoe in Ruhe.

        Diagnose: Akute Bronchitis, wahrscheinlich bakteriell bedingt.

        Therapie: Amoxicillin 1000 Milligramm, dreimal täglich für sieben Tage.
        ACC 600 Milligramm einmal täglich zur Schleimlösung.
        Paracetamol 500 Milligramm bei Bedarf gegen Fieber.

        Kontrolluntersuchung in einer Woche empfohlen.
        Arbeitsunfähigkeitsbescheinigung für fünf Tage ausgestellt.
        """
    },
    {
        "filename": "02_hypertonie.mp3",
        "text": """
        Patient Schmidt, Thomas, zweiundsechzig Jahre alt.
        Kontrolluntersuchung wegen arterieller Hypertonie.

        Patient berichtet über gelegentliche Kopfschmerzen morgens.
        Keine Schwindel oder Sehstörungen.

        Befunde: Blutdruck heute 165 zu 95 Millimeter Quecksilbersäule.
        Im Durchschnitt der letzten Messungen 160 zu 90.
        Puls 76, regelmäßig. Herzauskultation unauffällig.
        EKG zeigt Sinusrhythmus ohne pathologische Veränderungen.

        Diagnose: Arterielle Hypertonie Grad 2, derzeit nicht optimal eingestellt.

        Therapieplan: Ramipril Dosis erhöhen auf 10 Milligramm einmal täglich morgens.
        Zusätzlich Amlodipin 5 Milligramm einmal täglich.

        Empfehlungen: Salzarme Ernährung, regelmäßige Bewegung mindestens 30 Minuten täglich.
        Blutdruckkontrolle zu Hause zweimal täglich dokumentieren.

        Wiedervorstellung in vier Wochen zur Kontrolle.
        Laborwerte: Nierenwerte und Elektrolyte in drei Wochen kontrollieren.
        """
    },
    {
        "filename": "03_gastritis.mp3",
        "text": """
        Patientin Weber, Anna, achtunddreißig Jahre alt.
        Beschwerden: Seit zwei Wochen brennende Schmerzen im Oberbauch, besonders nüchtern.
        Übelkeit am Morgen. Keine Erbrechen.

        Anamnese: Vermehrter Stress bei der Arbeit in den letzten Monaten.
        Regelmäßige Einnahme von Ibuprofen gegen Kopfschmerzen, etwa dreimal wöchentlich.

        Befunde: Abdomen weich, Druckschmerz im Epigastrium.
        Keine Abwehrspannung. Darmgeräusche normal.

        Diagnose: Verdacht auf akute Gastritis, möglicherweise medikamentös induziert durch Ibuprofen.

        Maßnahmen: Ibuprofen sofort absetzen.
        Pantoprazol 40 Milligramm einmal täglich morgens vor dem Frühstück für vier Wochen.
        Bei Bedarf Paracetamol statt Ibuprofen gegen Schmerzen.

        Zusätzliche Hinweise: Stress-Reduktion, regelmäßige Mahlzeiten, Vermeidung von scharfem Essen und Alkohol.

        Falls keine Besserung innerhalb von zwei Wochen, Gastroskopie erforderlich.
        Wiedervorstellung in zwei Wochen.
        """
    },
    {
        "filename": "04_diabetes_kontrolle.mp3",
        "text": """
        Patient Klein, Michael, fünfundfünfzig Jahre alt.
        Quartalskontrolle Diabetes mellitus Typ 2.

        Patient gibt an, Medikation regelmäßig einzunehmen.
        Blutzuckerselbstkontrolle zeigt Nüchternwerte zwischen 110 und 140 Milligramm pro Deziliter.

        Befunde: Gewicht heute 92 Kilogramm bei Körpergröße 175 Zentimeter.
        Body-Mass-Index 30,0. Leichte Gewichtszunahme von 2 Kilogramm seit letzter Kontrolle.

        Laborwerte: HbA1c-Wert 7,2 Prozent. Liegt über Zielwert von unter 7 Prozent.
        Nierenwerte im Normbereich. Keine Proteinurie.

        Diagnose: Diabetes mellitus Typ 2, derzeit suboptimale Stoffwechseleinstellung.

        Therapieanpassung: Metformin erhöhen auf 1000 Milligramm zweimal täglich zu den Mahlzeiten.

        Zusätzliche Maßnahmen: Dringend Gewichtsreduktion empfohlen, Ziel minus 5 Kilogramm in drei Monaten.
        Ernährungsberatung vereinbaren. Bewegung steigern auf täglich 30 Minuten.

        Nächste Kontrolle in drei Monaten mit HbA1c-Bestimmung.
        Augenarzt-Kontrolle dieses Jahr noch ausstehend, Überweisung ausgehändigt.
        """
    },
    {
        "filename": "05_asthma.mp3",
        "text": """
        Patientin Fischer, Julia, neunundzwanzig Jahre alt.
        Bekanntes allergisches Asthma bronchiale seit Kindheit.

        Aktuelle Beschwerden: Zunahme von Atemnot und Giemen in den letzten zwei Wochen.
        Besonders nachts und morgens. Notwendigkeit des Notfall-Sprays täglich mehrfach.

        Anamnese: Pollenflugzeit hat begonnen. Patientin vermutet Zusammenhang mit Birkenpollenallergie.

        Befunde: Auskultation ergibt exspiratorisches Giemen und verlängertes Exspirium.
        Peak-Flow-Messung 320 Liter pro Minute, Bestwert der Patientin 450 Liter pro Minute.
        Sauerstoffsättigung 96 Prozent.

        Diagnose: Asthma bronchiale, allergische Exazerbation.
        Unzureichende Asthmakontrolle.

        Therapie: Inhalatives Kortikosteroid erhöhen. Budesonid 400 Mikrogramm zweimal täglich.
        Zusätzlich Montelukast 10 Milligramm einmal abends als Tablette.
        Salbutamol-Spray bei Bedarf, maximal viermal täglich.

        Antihistaminikum: Loratadin 10 Milligramm einmal täglich während der Pollensaison.

        Peak-Flow-Protokoll zu Hause führen.
        Wiedervorstellung in zwei Wochen oder bei akuter Verschlechterung sofort.
        Bei schwerer Atemnot oder unzureichendem Ansprechen auf Notfall-Spray, Notaufnahme aufsuchen.
        """
    },
    {
        "filename": "06_rueckenschmerzen.mp3",
        "text": """
        Patient Becker, Andreas, zweiundvierzig Jahre alt.
        Erstvorstellung wegen akuter Rückenschmerzen.

        Beschwerden: Seit drei Tagen starke Schmerzen im unteren Rücken, linksseitig betont.
        Beginn nach schwerem Heben bei Umzugsarbeiten.
        Keine Ausstrahlung ins Bein. Keine Taubheitsgefühle.

        Befunde: Klopfschmerz über Lendenwirbelsäule Segment L4 bis L5.
        Bewegungseinschränkung der Lendenwirbelsäule, besonders bei Flexion.
        Lasègue-Zeichen negativ beidseits. Keine neurologischen Ausfälle.
        Sensibilität und Motorik der unteren Extremität intakt.

        Diagnose: Akutes Lumbalsyndrom, muskulär bedingt.
        Keine Hinweise auf Nervenwurzelkompression oder Bandscheibenvorfall.

        Therapie: Ibuprofen 600 Milligramm dreimal täglich zu den Mahlzeiten für fünf Tage.
        Tetrazepam 50 Milligramm abends zur Muskelrelaxation für drei Tage.

        Physiotherapie: Verordnung für manuelle Therapie, sechs Behandlungen.

        Empfehlungen: Stufenlagerung zur Entlastung. Wärmeanwendungen.
        Keine Bettruhe, sondern leichte Mobilisation soweit schmerzfrei möglich.
        Schweres Heben vermeiden für mindestens zwei Wochen.

        Arbeitsunfähigkeit für eine Woche.
        Wiedervorstellung bei fehlender Besserung nach fünf Tagen.
        """
    },
    {
        "filename": "07_harnwegsinfekt.mp3",
        "text": """
        Patientin Meyer, Sabine, einunddreißig Jahre alt.
        Akute Beschwerden seit gestern.

        Hauptbeschwerde: Brennen beim Wasserlassen, häufiger Harndrang.
        Gefühl der unvollständigen Blasenentleerung.
        Kein Fieber, keine Flankenschmerzen.

        Befunde: Unterbauch leicht druckschmerzhaft suprapubisch.
        Temperatur 36,8 Grad Celsius. Nierenlager beidseits frei.

        Urinstreifentest: Leukozyten positiv, drei plus. Nitrit positiv.
        Erythrozyten negativ. Keine Proteinurie.

        Diagnose: Akute unkomplizierte Zystitis, Harnwegsinfekt.

        Therapie: Fosfomycin-Trometamol 3 Gramm als Einmaldosis abends.

        Empfehlungen: Reichlich Flüssigkeit trinken, mindestens zwei Liter täglich.
        Blasen- und Nierentee unterstützend.
        Wärme auf Unterbauch kann lindernd wirken.

        Wichtig: Beim Auftreten von Fieber oder Flankenschmerzen sofort wiedervorstellen.
        Kontrolluntersuchung nicht erforderlich bei prompter Besserung.
        Bei persistierenden Beschwerden nach drei Tagen Wiedervorstellung.
        Urinkultur dann erforderlich.
        """
    },
    {
        "filename": "08_angina.mp3",
        "text": """
        Patient Schneider, Max, sechsundzwanzig Jahre alt.
        Beschwerden seit zwei Tagen.

        Hauptbeschwerde: Starke Halsschmerzen beidseits, Schluckbeschwerden.
        Fieber bis 39 Grad Celsius. Abgeschlagenheit und Kopfschmerzen.

        Befunde: Inspektion des Rachens zeigt stark gerötete und geschwollene Tonsillen beidseits.
        Weißliche Stippchen auf beiden Tonsillen sichtbar.
        Zervikale Lymphknoten beidseits schmerzhaft geschwollen.
        Temperatur aktuell 38,4 Grad Celsius.

        Rachenabstrich für Streptokokken-Schnelltest: Positiv.

        Diagnose: Akute eitrige Tonsillitis, Streptokokken-Angina.

        Therapie: Penicillin V 1,5 Mega, dreimal täglich für zehn Tage.
        Wichtig: Antibiotikum vollständig einnehmen, auch bei Besserung der Symptome.

        Symptomatisch: Paracetamol 500 Milligramm bis zu viermal täglich gegen Fieber und Schmerzen.
        Lutschtabletten mit Lokalanästhetikum gegen Halsschmerzen.

        Allgemeinmaßnahmen: Schonung, Bettruhe bei Fieber.
        Reichlich trinken, lauwarme Getränke bevorzugen.
        Weiche, nicht reizende Kost.

        Arbeitsunfähigkeit für eine Woche.
        Wiedervorstellung bei ausbleibendem Fieberrückgang nach drei Tagen.
        Ansteckungsgefahr besteht bis 24 Stunden nach Antibiotikabeginn.
        """
    }
]

# Generate audio files
print("Generating German medical audio samples...")
for i, scenario in enumerate(scenarios, 1):
    print(f"[{i}/{len(scenarios)}] Creating {scenario['filename']}...")

    try:
        # Create gTTS object
        tts = gTTS(text=scenario['text'].strip(), lang='de', slow=False)

        # Save audio file
        output_path = output_dir / scenario['filename']
        tts.save(str(output_path))

        print(f"    ✓ Saved to {output_path}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

print(f"\n✅ Generated {len(scenarios)} test audio files in {output_dir}/")
print("\nTest scenarios:")
print("1. Bronchitis - Respiratory infection with cough")
print("2. Hypertonie - High blood pressure control visit")
print("3. Gastritis - Stomach inflammation")
print("4. Diabetes Kontrolle - Diabetes quarterly check")
print("5. Asthma - Allergic asthma exacerbation")
print("6. Rückenschmerzen - Acute lower back pain")
print("7. Harnwegsinfekt - Urinary tract infection")
print("8. Angina - Streptococcal throat infection")
