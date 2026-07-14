from mix_match import mix_match_class

mm = mix_match_class()
mixes = []
for i in range(50) :
    mixes.append({'HSR' : [(3000, 0)], 'Noamp' : [(1000, 0)]})
# print(mm.example(methods = [("KNN", "KNN", {}), ("GMM", "GMM", {})], show_hists= False))

# mixes = []
# for i in range(50) :
#     mixes.append({'ecDNA' : [(80, 0)], 'Noamp' : [(3920, 0)]})

mixes = []
for i in range(50) :
    mixes.append({'ecDNA' : [(20, 0)], 'HSR' : [(480, 0)]})

mixes = []
for i in range(50) :
    mixes.append({ 'HSR' : [(480, 0)]})
mm.set_mixes(mixes)
# mixes = []
# for i in range(50) :
#     mixes.append({'HSR' : [(1920, 0)], 'Noamp' : [(1920, 0)], 'ecDNA' : [(160, 0)]})
# mm.set_mixes(mixes)

# print(mm.calculate_mixes(methods = [("GMM", "GMM", {"kneedle_coeff" : 3})], show_hists=False))
print(mm.calculate_mixes(methods = [("KNN", "KNN", {"k_mult" : 1})], show_hists=False))

print("Done")


