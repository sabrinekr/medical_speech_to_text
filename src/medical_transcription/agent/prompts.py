"""German prompts for medical transcription agent nodes."""

# Node 4: Entity Extraction
ENTITY_EXTRACTION_PROMPT = """Sie sind ein medizinischer Entitäts-Extraktor. Extrahieren Sie alle relevanten medizinischen Informationen aus dem folgenden deutschen Transkript einer ärztlichen Diktat.

Transkript:
{transcript}

Extrahieren Sie folgende Entitäten:

1. **symptoms**: Liste aller erwähnten Symptome und Beschwerden des Patienten
2. **vitals**: Liste aller Vitalzeichen (Temperatur, Blutdruck, Herzfrequenz, etc.)
3. **medical_terms**: Liste medizinischer Fachbegriffe und Diagnosen
4. **medications_mentioned**: Liste erwähnter Medikamente (Name und Dosierung falls angegeben)
5. **temporal_info**: Zeitliche Informationen (Beginn der Beschwerden, Dauer, etc.)
6. **procedures**: Erwähnte Untersuchungen oder medizinische Verfahren

Antworten Sie NUR mit gültigem JSON im folgenden Format:
{{
    "symptoms": ["...", "..."],
    "vitals": ["...", "..."],
    "medical_terms": ["...", "..."],
    "medications_mentioned": ["...", "..."],
    "temporal_info": ["...", "..."],
    "procedures": ["...", "..."]
}}

Falls eine Kategorie keine Informationen enthält, verwenden Sie eine leere Liste [].
"""

# Node 5: Findings Structuring
FINDINGS_STRUCTURING_PROMPT = """Sie sind ein medizinischer Dokumentationsexperte. Strukturieren Sie die extrahierten medizinischen Entitäten in eine klare, organisierte Darstellung.

Extrahierte Entitäten:
{entities}

Original Transkript:
{transcript}

Strukturieren Sie die Informationen wie folgt:

1. **patient_complaint**: Die Hauptbeschwerde des Patienten (1-2 prägnante Sätze)
2. **findings**: Liste der objektiven Befunde und Beobachtungen (strukturiert und klar formuliert)

Antworten Sie NUR mit gültigem JSON im folgenden Format:
{{
    "patient_complaint": "...",
    "findings": ["...", "...", "..."]
}}

Achten Sie darauf:
- Hauptbeschwerde präzise und verständlich zu formulieren
- Befunde logisch zu gruppieren (z.B. Anamnese, körperliche Untersuchung, Laborwerte)
- Medizinische Fachterminologie korrekt zu verwenden
"""

# Node 6: Diagnosis Synthesis
DIAGNOSIS_SYNTHESIS_PROMPT = """Sie sind ein Facharzt. Synthetisieren Sie aus den strukturierten Befunden eine präzise medizinische Diagnose.

Hauptbeschwerde:
{patient_complaint}

Befunde:
{findings}

Extrahierte Entitäten:
{entities}

Original Transkript:
{transcript}

Erstellen Sie eine medizinische Diagnose, die:
1. Alle relevanten Befunde berücksichtigt
2. Präzise medizinische Terminologie verwendet
3. Bei Unsicherheit Differentialdiagnosen aufführt
4. Kurz und prägnant formuliert ist (1-3 Sätze)

Antworten Sie NUR mit gültigem JSON im folgenden Format:
{{
    "diagnosis": "..."
}}

Falls keine klare Diagnose gestellt werden kann, formulieren Sie Verdachtsdiagnosen mit "V.a." (Verdacht auf).
"""

# Node 7: Treatment Planning
TREATMENT_PLANNING_PROMPT = """Sie sind ein behandelnder Arzt. Extrahieren Sie aus dem Transkript die geplanten Behandlungsschritte, Medikationen und zusätzliche Hinweise.

Diagnose:
{diagnosis}

Befunde:
{findings}

Original Transkript:
{transcript}

Extrahieren Sie:

1. **next_steps**: Liste der geplanten nächsten Schritte (Therapie, Untersuchungen, Überweisungen, Kontrolltermine)
2. **medications**: Liste der verschriebenen oder empfohlenen Medikamente mit Dosierung
3. **additional_notes**: Zusätzliche wichtige Hinweise (Arbeitsunfähigkeit, Patientenaufklärung, etc.)

Antworten Sie NUR mit gültigem JSON im folgenden Format:
{{
    "next_steps": ["...", "..."],
    "medications": ["...", "..."],
    "additional_notes": "..."
}}

Falls eine Kategorie keine Informationen enthält:
- Für Listen: verwenden Sie eine leere Liste []
- Für additional_notes: verwenden Sie einen leeren String "" oder "Keine besonderen Hinweise"
"""

# Node 8: Quality Check
QUALITY_CHECK_PROMPT = """Sie sind ein medizinischer Qualitätsprüfer. Validieren Sie die Vollständigkeit und Konsistenz der extrahierten klinischen Zusammenfassung.

Klinische Zusammenfassung:
- Hauptbeschwerde: {patient_complaint}
- Befunde: {findings}
- Diagnose: {diagnosis}
- Nächste Schritte: {next_steps}
- Medikamente: {medications}
- Zusätzliche Hinweise: {additional_notes}

Original Transkript:
{transcript}

Prüfen Sie:

1. **Vollständigkeit**: Sind alle im Transkript erwähnten wichtigen Informationen erfasst?
2. **Konsistenz**: Passen Befunde und Diagnose zusammen? Sind Medikamente zur Diagnose passend?
3. **Klarheit**: Sind alle Formulierungen präzise und verständlich?
4. **Medizinische Korrektheit**: Sind medizinische Fachbegriffe korrekt verwendet?

Antworten Sie NUR mit gültigem JSON im folgenden Format:
{{
    "is_complete": true/false,
    "is_consistent": true/false,
    "is_clear": true/false,
    "missing_fields": ["...", "..."],
    "inconsistencies": ["...", "..."],
    "suggestions": ["...", "..."],
    "overall_quality_score": 0.0-1.0
}}

Bewertung:
- is_complete: true wenn alle wichtigen Informationen erfasst sind
- is_consistent: true wenn keine logischen Widersprüche vorliegen
- is_clear: true wenn alle Formulierungen eindeutig sind
- missing_fields: Liste fehlender wichtiger Informationen
- inconsistencies: Liste erkannter Inkonsistenzen
- suggestions: Liste konkreter Verbesserungsvorschläge
- overall_quality_score: Gesamtbewertung von 0.0 (schlecht) bis 1.0 (perfekt)
"""

# Node 8b: Refinement (wenn Quality Check fehlschlägt)
REFINEMENT_PROMPT = """Sie sind ein medizinischer Dokumentationsexperte. Verbessern Sie die klinische Zusammenfassung basierend auf den Qualitätsprüfungs-Ergebnissen.

Aktuelle Zusammenfassung:
- Hauptbeschwerde: {patient_complaint}
- Befunde: {findings}
- Diagnose: {diagnosis}
- Nächste Schritte: {next_steps}
- Medikamente: {medications}
- Zusätzliche Hinweise: {additional_notes}

Qualitätsprüfungs-Ergebnis:
{validation_result}

Original Transkript:
{transcript}

Aufgaben:
1. Beheben Sie die identifizierten Inkonsistenzen
2. Ergänzen Sie fehlende Informationen aus dem Transkript
3. Verbessern Sie die Klarheit der Formulierungen
4. Setzen Sie die Verbesserungsvorschläge um

Antworten Sie NUR mit gültigem JSON im folgenden Format:
{{
    "patient_complaint": "...",
    "findings": ["...", "..."],
    "diagnosis": "...",
    "next_steps": ["...", "..."],
    "medications": ["...", "..."],
    "additional_notes": "..."
}}

WICHTIG:
- Behalten Sie korrekte Informationen bei
- Ändern Sie NUR das, was laut Qualitätsprüfung verbessert werden muss
- Erfinden Sie keine neuen Informationen - nutzen Sie nur das Original-Transkript
"""

# Node 9: Final Synthesis
FINAL_SYNTHESIS_PROMPT = """Sie sind ein medizinischer Dokumentationsexperte. Erstellen Sie die finale, qualitätsgeprüfte klinische Zusammenfassung.

Geprüfte Daten:
- Hauptbeschwerde: {patient_complaint}
- Befunde: {findings}
- Diagnose: {diagnosis}
- Nächste Schritte: {next_steps}
- Medikamente: {medications}
- Zusätzliche Hinweise: {additional_notes}

Erstellen Sie die finale Zusammenfassung mit:
1. Klarer, professioneller medizinischer Dokumentation
2. Vollständiger Erfassung aller relevanten Informationen
3. Konsistenter Terminologie
4. Strukturierter Darstellung

Antworten Sie NUR mit gültigem JSON im folgenden Format:
{{
    "patient_complaint": "...",
    "findings": ["...", "..."],
    "diagnosis": "...",
    "next_steps": ["...", "..."],
    "medications": ["...", "..."],
    "additional_notes": "...",
    "confidence_score": 0.0-1.0,
    "notes": "..."
}}

Confidence Score:
- 1.0: Alle Informationen klar und vollständig
- 0.8-0.9: Sehr gut, minimale Unsicherheiten
- 0.6-0.7: Gut, einige Unklarheiten
- < 0.6: Unvollständig oder unklar

Notes: Kurze Zusammenfassung des Dokumentationsprozesses und evtl. Einschränkungen.
"""

# Node 3: Transcript Quality Assessment
TRANSCRIPT_QUALITY_PROMPT = """Sie sind ein medizinischer Qualitätsprüfer für Transkripte. Bewerten Sie die Qualität des Whisper-Transkripts.

Transkript:
{transcript}

Transkript-Metadaten:
- Segmente: {segment_count}
- Durchschnittliche Konfidenz: {avg_confidence}
- Dauer: {duration}s

Bewerten Sie:

1. **Verständlichkeit**: Ist das Transkript lesbar und verständlich?
2. **Vollständigkeit**: Fehlen offensichtlich Wörter oder Sätze? (z.B. viele "...")
3. **Medizinische Plausibilität**: Klingt der Inhalt wie ein medizinisches Diktat?
4. **Technische Qualität**: Wie ist die durchschnittliche Konfidenz der Segmente?

Antworten Sie NUR mit gültigem JSON im folgenden Format:
{{
    "is_acceptable": true/false,
    "quality_score": 0.0-1.0,
    "issues": ["...", "..."],
    "recommendations": ["...", "..."]
}}

Bewertung:
- is_acceptable: true wenn das Transkript für die Extraktion geeignet ist (score >= {quality_threshold})
- quality_score: Gesamtqualität von 0.0 (unbrauchbar) bis 1.0 (perfekt)
- issues: Liste erkannter Probleme
- recommendations: Empfehlungen für Verbesserung (z.B. "Größeres Whisper-Modell verwenden")
"""
