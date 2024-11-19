import numpy as np
import csv

proj_psnr = []
proj_ssim = []
proj_time = []

with open('resultmetrics', 'r') as file: #split file entires into individual metrics
    entries = csv.reader(file)

    for row in entries:
        print(f"Entries: {row}")
        for i in range(0, len(row), 1):
            if i % 3 == 0:
                proj_ssim.append(float(row[i].strip()))
            elif i % 3 == 1:
                proj_psnr.append(float(row[i].strip()))
            elif i % 3 == 2:
                proj_time.append(float(row[i].strip()))

summy = 0
for entry in proj_ssim:
    summy += entry
average_ssim_ = summy / len(proj_ssim)

summy_psnr = 0
for entry in proj_psnr:
    summy_psnr += entry
average_psnr_ = summy_psnr / len(proj_psnr)

summy_time = 0
for entry in proj_time:
    summy_time += entry
average_time_ = summy_time / len(proj_time)

deviation_ssim_ = np.std(proj_ssim)
deviation_psnr_ = np.std(proj_psnr)
deviation_time_ = np.std(proj_time)

print(f"Average SSIM proj: {average_ssim_:.5f}, Deviation SSIM proj: {deviation_ssim_:.5f}")
print(f"Average PSNR proj: {average_psnr_:.5f}, Deviation PSNR proj: {deviation_psnr_:.5f}")
print(f"Average TIME proj: {average_time_:.5f}, Deviation TIME proj: {deviation_time_:.5f}")

'''
history for reuse maybe
'''
#float16 0-9
# proj_psnr = [34.77770804766121,35.06133423415108,34.873976196268,34.8085189421117,34.865565419423476,35.10073782564656,35.051417281581635,34.23267659668869]
# proj_ssim = [0.7977958487326878,0.817332111321532,0.8038475466565188,0.7983933602641442,0.8019470349258365,0.8140416844430352,0.8031349792443101,0.8124636464741591,0.7592103502737222]
# proj_time = [1.025866484642029,0.9795293688774109,1.0344816088676452,1.0318101922671,0.9815725803375244,0.8681439677874248,1.0681977033615113,0.9863366882006327]

#baseline 0-9
# proj_psnr = [35.218837094252805,35.06413459528928,34.96155526208463,35.219192551128174,35.137227436453614,35.20159714698291,35.22975980413602,36.265243208156576,36.114736956878176]
# proj_ssim = [0.7474276528633433,0.7441132082429928,0.741236928404448,0.7473786317645884,0.7464334015657358,0.7459497708278342,0.7467006485795611,0.8077076615478447,0.8572814834817941]
# proj_time = [1.137521262963613,1.143937095006307,1.138495719432831,1.1352513511975606,1.1337273915608723,1.1335405826568603,1.1890284498532613,1.130593510468801,1.1420231739679971]
