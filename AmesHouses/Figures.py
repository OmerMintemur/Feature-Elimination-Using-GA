import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline

sns.set_theme()

data = pd.read_csv("Results_Pop1000_Gen_20.csv")

# fig = plt.figure(figsize=(12,8))
# # plt.subplot(1, 2, 1)
# x = np.arange(10)
# X_Y_Spline = make_interp_spline(x, data["BestFitness"])
# X_ = np.linspace(x.min(), x.max(), 500)
# Y_ = X_Y_Spline(X_)
# plt.plot(X_, Y_,label="Best Fitness")
#
#
# # plt.subplot(1, 2, 2)
# x = np.arange(10)
# X_Y_Spline = make_interp_spline(x, data["AverageFitness"])
# X_ = np.linspace(x.min(), x.max(), 500)
# Y_ = X_Y_Spline(X_)
#
# plt.plot(X_, Y_,label="Average Fitness")
# plt.title("Average Fitness Values Per Generation")
# plt.xlabel("Number of Generation")
# plt.ylabel("Average Fitness Values")
# plt.legend()
# fig1 = plt.gcf()
# fig1.savefig('Results_Fig.eps', format='eps',dpi=600,bbox_inches="tight")
# plt.show()


# data['FeaturesLength'] = data['SelectedFeatures'].apply(len)
# print(data['FeaturesLength'])

data['FeaturesLength'] = data['SelectedFeatures'].str.strip(' ,[]').str.count(',') + 1
print (data['FeaturesLength'])


x = np.arange(10)
X_Y_Spline = make_interp_spline(x, data["FeaturesLength"])
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)

plt.plot(X_, Y_)
plt.title("Number of Features Per Generation")
plt.xlabel("Number of Generation")
plt.ylabel("Number of Features")
fig1 = plt.gcf()
fig1.savefig('Results_Fig_Features.eps', format='eps',dpi=600)
plt.show()