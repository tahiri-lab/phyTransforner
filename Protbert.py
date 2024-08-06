from transformers import pipeline

# Créer un pipeline de classification avec le modèle ProtBERT
pipe = pipeline("text-classification", model="Rocketknight1/esm2_t6_8M_UR50D-finetuned-localization")

# Dictionnaire des séquences de protéines par localisation subcellulaire
protein_sequences = {
    'Cell.membrane': "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFK",
    'Cytoplasm': "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVG",
    'Endoplasmic.reticulum': "MKAAVRKVLTVLLLAAAVAGCGNASAEANQNGKPR",
    'Extracellular': "MGRDGIDTDVFSGPDGKTGQSINNYGGFGADND",
    'Golgi.apparatus': "MKSVLLLALSLWILPGGQVTQGVDLSSFGNSDLK",
    'Lysosome/Vacuole': "MKTLLLAILAAWATAEAQTAAPCSGSADAAPTP",
    'Mitochondrion': "MALWMRLLPLLALLALWGPGPGLSGLALLLAVAP",
    'Nucleus': "MGLRSGRGKTGGKARAKAKSRSSRAGLQFPVGR",
    'Peroxisome': "MNLREVRDPLPAHLGRFLRVAAAYRLARFGSD",
    'Plastid': "MSTIAHRAMVALGEPNAETMGRLEREGAEVRN"
}

# Dictionnaire des séquences de protéines avec uniquement la localisation "Mitochondrion"
mitochondrion_sequences = {
    'Mitochondrion_1': "MALWMRLLPLLALLALWGPGPGL",
    'Mitochondrion_2': "MLAKKKPQKPLLPLTPEELPAELTDLT",
    'Mitochondrion_3': "MTSKRSKAAVRRLAAAAAAPVG",
    'Mitochondrion_4': "MVKALWLLPLALLAVQLAHAAG",
    'Mitochondrion_5': "MSWKTLLPLVAFALSVTAFS",
}

# Tester chaque séquence avec le modèle et imprimer les résultats
for location, sequence in mitochondrion_sequences.items():
    result = pipe(sequence)
    print(f"Expected: {location}, Predicted: {result[0]['label']}, Score: {result[0]['score']:.4f}")
