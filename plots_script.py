from mix_match import mix_match_class
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

###############################################################

# Remaining are all Noamp
def create_mix_noamp(NO_ITER, max_ecDNA = 1000, total_cells = 4000) :
    mixes = []

    ecDNA_num_float = 0.0
    while ecDNA_num_float <= max_ecDNA :
        ecDNA_num = round(ecDNA_num_float)
        mix = {}
        Noamp_num = total_cells - ecDNA_num
        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 0)]
        
        for i in range(NO_ITER) :
            mixes.append(mix)

        ecDNA_num_float += total_cells/100 
    return mixes

# Remaining are all HSR
def create_mix_HSR(NO_ITER, max_ecDNA = 800, total_cells = 4000) :
    mixes = []

    ecDNA_num_float = 0.0
    while ecDNA_num_float <= max_ecDNA :
        ecDNA_num = round(ecDNA_num_float)
        mix = {}
        HSR_num = total_cells - ecDNA_num
        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
        
        for it in range(NO_ITER) :
            mixes.append(mix)
        ecDNA_num_float += total_cells/100 

    return mixes

# Half HSR, half Noamp
def create_mix_HSR_noamp(NO_ITER, max_ecDNA = 800, total_cells = 4000) :
    mixes = []
    
    ecDNA_num_float = 0.0
    while ecDNA_num_float <= max_ecDNA :
        ecDNA_num = round(ecDNA_num_float)
        mix = {}
        Noamp_num = int((total_cells - ecDNA_num)/2)
        HSR_num = int((total_cells - ecDNA_num)/2)

        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 0)]
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
        for it in range(NO_ITER) :
            mixes.append(mix)
        ecDNA_num_float += total_cells/100 

    return mixes

# HSR mimics ecDNA mean (distributions overlap)
def create_mix_HSR_p10(NO_ITER, max_ecDNA = 1600, total_cells = 4000) :
    mixes = []

    ecDNA_num_float = 0.0
    while ecDNA_num_float <= max_ecDNA :
        ecDNA_num = round(ecDNA_num_float)
        mix = {}
        HSR_num = int(total_cells - ecDNA_num)

        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
    
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 10)]
        for it in range(NO_ITER) :
            mixes.append(mix)
        ecDNA_num_float += total_cells/100 

    return mixes

# Just HSR and Noamp
def create_HSR_noamp(NO_ITER, max_HSR = 4000, total_cells = 4000) :
    mixes = []

    HSR_num_float = 0.0
    while HSR_num_float <= max_HSR :
        HSR_num = round(HSR_num_float)
        mix = {}
        Noamp_num = int(total_cells - HSR_num)

        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
    
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 10)]
        for it in range(NO_ITER) :
            mixes.append(mix)
        HSR_num_float += total_cells/100 

    return mixes

#################################################################

NO_ITER = 20
methods = [('GMM', 'GMM', {}), ('scAmp', 'NN', {}), ('KNN', 'KNN', {})]

# {'Method' : (CREATE MIX FUNCTION, {ARGS}), 
# 'Name' : PLOT NAME, 
# 'Save' : PLOT SAVE LOCATION,
# 'ecDNA' : ecDNA LOCATION, 
# 'HSR' : HSR LOCATION, 
# 'Noamp' : NOAMP LOCATION}
settings_list = [{'Method' : (create_mix_HSR, {"total_cells" : 1000, "max_ecDNA" : 400}),
                  'Name' : "ecDNA vs HSR (1000 Cells)",
                  'Save' : "./plots/ecDNA_HSR_1000.png",
                  'ecDNA' : "./vectors/COLO_MYC_ecDNA.tsv",
                  'HSR' : "./vectors/COLO_MYC_HSR.tsv",
                  'Noamp' : "./vectors/COLO_GAPDH_Noamp.tsv"}
                  ,
                  {'Method' : (create_mix_HSR, {"total_cells" : 500, "max_ecDNA" : 200}),
                  'Name' : "ecDNA vs HSR (500 Cells)",
                  'Save' : "./plots/ecDNA_HSR_500.png",
                  'ecDNA' : "./vectors/COLO_MYC_ecDNA.tsv",
                  'HSR' : "./vectors/COLO_MYC_HSR.tsv",
                  'Noamp' : "./vectors/COLO_GAPDH_Noamp.tsv"}
                  ,
                  {'Method' : (create_HSR_noamp, {}),
                  'Name' : "HSR vs Noamp (4000 Cells)",
                  'Save' : "./plots/HSR_Noamp_4000.png",
                  'ecDNA' : "./vectors/COLO_MYC_ecDNA.tsv",
                  'HSR' : "./vectors/COLO_MYC_HSR.tsv",
                  'Noamp' : "./vectors/COLO_GAPDH_Noamp.tsv"}
                  ,
                #   {'Method' : (create_mix_noamp, {}),
                #   'Name' : "ecDNA vs Noamp (4000 Cells)",
                #   'Save' : "./plots/ecDNA_noAmp_4000.png",
                #   'ecDNA' : "./vectors/COLO_MYC_ecDNA.tsv",
                #   'HSR' : "./vectors/COLO_MYC_HSR.tsv",
                #   'Noamp' : "./vectors/COLO_GAPDH_Noamp.tsv"}
                #   ,
                #   {'Method' : (create_mix_HSR, {}),
                #   'Name' : "ecDNA vs HSR (4000 Cells)",
                #   'Save' : "./plots/ecDNA_HSR_4000.png",
                #   'ecDNA' : "./vectors/COLO_MYC_ecDNA.tsv",
                #   'HSR' : "./vectors/COLO_MYC_HSR.tsv",
                #   'Noamp' : "./vectors/COLO_GAPDH_Noamp.tsv"}
                #   ,
                #   {'Method' : (create_mix_HSR_noamp, {}),
                #   'Name' : "ecDNA vs Half HSR, Half Noamp (4000 Cells)",
                #   'Save' : "./plots/ecDNA_half_4000.png",
                #   'ecDNA' : "./vectors/COLO_MYC_ecDNA.tsv",
                #   'HSR' : "./vectors/COLO_MYC_HSR.tsv",
                #   'Noamp' : "./vectors/COLO_GAPDH_Noamp.tsv"}
                #   ,
                #   {'Method' : (create_mix_HSR, {}),
                #   'Name' : "ecDNA vs HSR + 10 (4000 Cells)",
                #   'Save' : "./plots/ecDNA_HSRp10_4000.png",
                #   'ecDNA' : "./vectors/COLO_MYC_ecDNA.tsv",
                #   'HSR' : "./vectors/COLO_MYC_HSR.tsv",
                #   'Noamp' : "./vectors/COLO_GAPDH_Noamp.tsv"}
                  ]



for settings in settings_list :
    mm = mix_match_class(HSR_data=[(settings['HSR'], 'HSR')], ecDNA_data=[(settings['ecDNA'], 'ecDNA')], Noamp_data=[(settings['Noamp'],'Noamp')])

    mixes = settings['Method'][0](NO_ITER, **settings['Method'][1])
    mm.set_mixes(mixes)
    results = mm.calculate_mixes(methods = methods)

    for y, (name, vals) in enumerate(results.items()):
        vals = np.asarray(vals)
        vals = vals.reshape(-1, NO_ITER).mean(axis=1) * 100
        xs = range(0, len(vals))

        plt.plot(xs, vals, label = name, marker='o', alpha = 0.3)


    plt.title(settings['Name'])
    plt.legend()
    if settings['Method'][0].__name__ != "create_HSR_noamp" :
        plt.xlabel("ecDNA Percentage")
    else :
        plt.xlabel("HSR Percentage")

    plt.ylabel("Percent Predicted ecDNA")
    plt.tight_layout()
    plt.savefig(settings['Save'])
    plt.close()

