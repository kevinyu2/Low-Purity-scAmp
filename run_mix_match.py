from mix_match import mix_match_class

mm = mix_match_class()
mixes = []
for i in range(50) :
    mixes.append({'HSR' : [(2000, 0)], 'Noamp' : [(2000, 0)]})
# print(mm.example(methods = [("KNN", "KNN", {}), ("GMM", "GMM", {})], show_hists= False))
mm.set_mixes(mixes)
mixes = []
for i in range(50) :
    mixes.append({'HSR' : [(1920, 0)], 'Noamp' : [(1920, 0)], 'ecDNA' : [(160, 0)]})
mm.set_mixes(mixes)

print(mm.calculate_mixes(methods = [("GMM", "GMM", {})], show_hists=False))
print("Done")


