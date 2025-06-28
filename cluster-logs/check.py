import re

# Load the text
with open('/scratch/mt/new-structure/experiments/msaeed/masters/SCA/cluster-logs/lca2-77838.out', 'r') as file:
    content = file.read()

# Extract values using regular expressions
ssim_values = [float(x) for x in re.findall(r'SSIM is: ([\d.]+)', content)]
fid_values = [float(x) for x in re.findall(r'FID is: ([\d.]+)', content)]
psnr_values = [float(x) for x in re.findall(r'PSNR is: ([\d.]+)', content)]

print("SSIM:", sum(ssim_values)/len(ssim_values),len(ssim_values))
print("FID:", sum(fid_values)/len(fid_values),len(fid_values))
print("PSNR:", sum(psnr_values)/len(psnr_values),len(psnr_values))

print("***",float('nan'))