import pandas as pd
import random

def generate_demographic_data():
    wilayas = [
        "01 - Adrar", "02 - Chlef", "03 - Laghouat", "04 - Oum El Bouaghi", "05 - Batna",
        "06 - Béjaïa", "07 - Biskra", "08 - Béchar", "09 - Blida", "10 - Bouira",
        "11 - Tamanrasset", "12 - Tébessa", "13 - Tlemcen", "14 - Tiaret", "15 - Tizi Ouzou",
        "16 - Alger", "17 - Djelfa", "18 - Jijel", "19 - Sétif", "20 - Saïda",
        "21 - Skikda", "22 - Sidi Bel Abbès", "23 - Annaba", "24 - Guelma", "25 - Constantine",
        "26 - Médéa", "27 - Mostaganem", "28 - M'Sila", "29 - Mascara", "30 - Ouargla",
        "31 - Oran", "32 - El Bayadh", "33 - Illizi", "34 - Bordj Bou Arréridj", "35 - Boumerdès",
        "36 - El Tarf", "37 - Tindouf", "38 - Tissemsilt", "39 - El Oued", "40 - Khenchela",
        "41 - Souk Ahras", "42 - Tipaza", "43 - Mila", "44 - Aïn Defla", "45 - Naâma",
        "46 - Aïn Témouchent", "47 - Ghardaïa", "48 - Relizane"
    ]
    
    data = []
    for wilaya in wilayas:
        population = random.randint(200000, 3500000)
        area_km2 = random.randint(1000, 500000)
        density = population / area_km2
        data.append({
            "wilaya_code": wilaya.split(" - ")[0],
            "wilaya_name": wilaya.split(" - ")[1],
            "population_2024": population,
            "area_km2": area_km2,
            "density_per_km2": round(density, 2),
            "urbanization_rate": round(random.uniform(0.4, 0.95), 2)
        })
    
    df = pd.DataFrame(data)
    df.to_csv("demographic_sample_data.csv", index=False)
    print("Generated demographic_sample_data.csv")

if __name__ == "__main__":
    generate_demographic_data()
