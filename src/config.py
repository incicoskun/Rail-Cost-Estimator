from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
CSV_PATH       = DATA_PROCESSED / "global_rail_processed.csv"

FEATURE_ALL = [
    "country_te", "country_freq",
    "city_te", "city_freq",
    "tunnel_pct", "station_density",
    "log_length", "is_regional_rail",
    "mid_year"
]

PARAMS = dict(
    n_estimators=1200, 
    learning_rate=0.01, 
    max_depth=3, 
    subsample=0.8, 
    min_samples_leaf=5, 
    random_state=42
)

UI_COLORS = {
    "primary": "#4F46E5",     
    "primary_hover": "#4338CA",  
    "bg_light": "#F9FAFB",      
    "background": "#F9FAFB",   
    "card": "#FFFFFF",           
    "text_main": "#111827",   
    "text_muted": "#6B7280",   
    "accent_bg": "#EEF2FF",    
    "border": "#E5E7EB"          
}

CITY_MAP = {
    "AE": ["Dubai"], "AR": ["Buenos Aires"], "AT": ["Vienna"],
    "AU": ["Melbourne", "Perth", "Sydney"], "BD": ["Dhaka"], "BE": ["Brussels"],
    "BG": ["Sofia"], "BH": ["Bahrain"], "BR": ["Rio de Janeiro", "Salvador", "Sao Paulo"],
    "CA": ["Calgary", "Edmonton", "Hamilton", "Mississauga", "Montreal", "Ottawa", "Toronto", "Vancouver"],
    "CH": ["Geneva", "Lucerne", "Zurich"], "CL": ["Santiago"],
    "CN": ["Beijing", "Changchun", "Changsha", "Changzhou", "Chengdu", "Chongqing",
           "Dalian", "Dongguan", "Foshan", "Fuzhou", "Guangzhou", "Guilin", "Guiyang",
           "Hangzhou", "Harbin", "Hefei", "Hohhot", "Jiaxing", "Jinan", "Kunming",
           "Lanzhou", "Luoyang", "Nanchang", "Nanjing", "Nanning", "Nantong", "Ningbo",
           "Putian", "Qingdao", "Shanghai", "Shaoxing", "Shenyang", "Shenzhen",
           "Shijiazhuang", "Suzhou", "Taiyuan", "Taizhou", "Tianjin", "Urumqi",
           "Wenzhou", "Wuhan", "Wuhu", "Wuxi", "Xi'an", "Xiamen", "Xuzhou", "Zhengzhou", "Zibo"],
    "CO": ["Bogotá"], "CZ": ["Prague"],
    "DE": ["Berlin", "Cologne", "Dusseldorf", "Frankfurt", "Hamburg", "Karlsruhe", "Leipzig", "Munich", "Nuremberg"],
    "DK": ["Copenhagen"], "DR": ["Santo Domingo"], "EC": ["Quito"], "EG": ["Cairo"],
    "ES": ["Barcelona", "Bilbao", "Madrid", "Malaga", "Seville"], "FI": ["Helsinki"],
    "FR": ["Lyon", "Paris", "Rennes", "Toulouse"],
    "GR": ["Athens", "Thessaloniki"], "HK": ["Hong Kong"], "HU": ["Budapest"],
    "ID": ["Jakarta"], "IL": ["Tel Aviv"],
    "IN": ["Agra", "Ahmadabad", "Bangalore", "Chennai", "Delhi", "Gurgaon", "Hyderabad",
           "Jaipur", "Kanpur", "Kochi", "Lucknow", "Mumbai", "Nagpur", "Patna", "Pune", "Surat"],
    "IR": ["Tehran"],
    "IT": ["Brescia", "Catania", "Genova", "Milan", "Naples", "Rome", "Turin"],
    "JP": ["Fukuoka", "Hiroshima", "Kobe", "Nagoya", "Osaka", "Sapporo", "Sendai", "Tokyo"],
    "KR": ["Busan", "Incheon", "Seoul"], "KW": ["Kuwait City"],
    "MX": ["Guadalajara", "Mexico City"], "MY": ["Kuala Lumpur"],
    "NL": ["Amsterdam"], "NO": ["Oslo"], "NZ": ["Auckland"], "PA": ["Panama City"],
    "PE": ["Lima"], "PH": ["Manila"], "PK": ["Lahore"], "PL": ["Warsaw", "Łódź"],
    "PT": ["Lisbon", "Porto"], "QA": ["Doha"],
    "RO": ["Bucharest", "Cluj-Napoca"], "RS": ["Belgrade"],
    "RU": ["Kazan", "Moscow", "N. Novgorod", "Samara", "St. Petersburg", "Yekaterinburg"],
    "SA": ["Ad Dammam", "Jeddah", "Riyadh"], "SE": ["Gothenburg", "Malmo", "Stockholm"],
    "SG": ["Singapore"], "TH": ["Bangkok", "Chiang Mai"],
    "TR": ["Adana", "Ankara", "Bursa", "Istanbul", "Izmir"],
    "TW": ["Kaohsiung", "Taichung", "Tainan", "Taipei"],
    "UA": ["Kharkiv", "Kyiv"], "UK": ["London"],
    "US": ["Boston", "Chicago", "Honolulu", "Los Angeles", "Miami", "New York",
           "Philadelphia", "San Francisco", "Seattle", "Washington"],
    "UZ": ["Tashkent"], "VN": ["Hanoi", "Ho Chi Minh City"],
}

APP_INFO = {
    "title": "Adaptive Rail Cost Estimator",
    "engine_name": "Gradient Boosting Regressor",
    "dataset_version": "946 Projects",
    "accuracy_metrics": "R² 0.512 · MdAPE 18.6%",
    "footer": "Galatasaray University - Computer Engineering Graduation Project - Inci Coskun"
}

ENCODING_PARAMS = {
    "country_m": 20,  
    "city_m": 30,    
}

UNCERTAINTY_BANDS = {
    "lower": 0.65,  
    "upper": 1.45   
}

UNITS = {
    "currency": "M$",
    "length": "km",
    "cost_unit": "M$/km"
}