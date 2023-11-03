# do2_1, do6_1, etc are DREAMOutput objects

xdata = np.append(do2_1.grid.t, np.add(do2_2.grid.t[1:], do2_1.grid.t[-1])) * 1000

temp2 = np.append(do2_1.eqsys.T_cold[:,9], do2_2.eqsys.T_cold[1:, 9])
temp6 = np.append(do6_1.eqsys.T_cold[:,9], do6_2.eqsys.T_cold[1:, 9])
temp10 = np.append(do10_1.eqsys.T_cold[:,9], do10_2.eqsys.T_cold[1:, 9])

tot2 = np.append(do2_1.eqsys.I_p.data, do2_2.eqsys.I_p.data[1:])
tot6 = np.append(do6_1.eqsys.I_p.data, do6_2.eqsys.I_p.data[1:])
tot10 = np.append(do10_1.eqsys.I_p.data, do10_2.eqsys.I_p.data[1:])

re2 = np.append(do2_1.eqsys.j_re.current(), do2_2.eqsys.j_re.current(t=slice(1, None)))
re6 = np.append(do6_1.eqsys.j_re.current(), do6_2.eqsys.j_re.current(t=slice(1, None)))
re10 = np.append(do10_1.eqsys.j_re.current(), do10_2.eqsys.j_re.current(t=slice(1, None)))

lw = 3

fig = plt.figure(figsize=[10,7])
plt.plot(xdata, temp2, label=r"T$_e$(2$\cdot 10^{24}$)", lw=lw)
plt.plot(xdata, temp6, label=r"T$_e$(6$\cdot 10^{24}$)", lw=lw)
plt.plot(xdata, temp10, label=r"T$_e$(10$^{25}$)", lw=lw)
#plt.legend(facecolor="white", loc=1)
ax1 = plt.gca()
ax1.set_xlabel("Idő [ms]")
ax1.set_ylabel('Hőmérséklet [eV]')
ax1.tick_params(axis='y')
ax1.set_facecolor('white')
ax1.grid()
ax1.set_yscale("log")

ax2 = ax1.twinx()
ax2.set_ylabel("Teljes plazmaáram [MA]\nés elfutóelektron-áram [MA]")
ax2.plot(xdata, tot2 / 1e6, label=r"I$_p$(2$\cdot 10^{24}$)", linestyle="-.", lw=lw)
ax2.plot(xdata, tot6 / 1e6, label=r"I$_p$(6$\cdot 10^{24}$)", linestyle="-.", lw=lw)
ax2.plot(xdata, tot10 / 1e6, label=r"I$_p$(10$^{25}$)", linestyle="-.", lw=lw)
ax2.plot(xdata, re2 / 1e6, 'tab:blue', label=r"I$_{RE}$(2$\cdot 10^{24}$)", linestyle="--", lw=lw)
ax2.plot(xdata, re6 / 1e6, 'tab:orange', label=r"I$_{RE}$(6$\cdot 10^{24}$)", linestyle="--", lw=lw)
ax2.plot(xdata, re10 / 1e6, 'tab:green', label=r"I$_{RE}$(10$^{25}$)", linestyle="--", lw=lw)
#ax2.legend(facecolor="white",loc=3)
ax2.tick_params(axis='y')
#ax2.grid()
ax2.set_ylim([-0.05, None])
ax1.set_ylim([1e0, None])
ax1.set_xlim([0, 31])
ax2.set_xlim([0, 31])

# This part id for making the legends

xdata = np.zeros(10)

temp2 = np.zeros(10)
temp6 = np.zeros(10)
temp10 = np.zeros(10)

tot2 = np.zeros(10)
tot6 = np.zeros(10)
tot10 = np.zeros(10)

re2 = np.zeros(10)
re6 = np.zeros(10)
re10 = np.zeros(10)

lw = 3

fig = plt.figure(figsize=[10,7])
plt.plot(xdata, temp2, label=r"T$_e$(2$\cdot 10^{24}$)", lw=lw)
plt.plot(xdata, temp6, label=r"T$_e$(6$\cdot 10^{24}$)", lw=lw)
plt.plot(xdata, temp10, label=r"T$_e$(10$^{25}$)", lw=lw)
plt.legend(facecolor="white", loc=1)
ax1 = plt.gca()
ax1.set_xlabel("Idő [ms]")
ax1.set_ylabel('Hőmérséklet [eV]')
ax1.tick_params(axis='y')
ax1.set_facecolor('white')
#ax1.grid()
ax1.set_yscale("log")

ax2 = ax1.twinx()
ax2.set_ylabel("Teljes plazmaáram [MA]\nés elfutóelektron-áram [MA]")
ax2.plot(xdata, tot2 / 1e6, label=r"I$_p$(2$\cdot 10^{24}$)", linestyle="-.", lw=lw)
ax2.plot(xdata, tot6 / 1e6, label=r"I$_p$(6$\cdot 10^{24}$)", linestyle="-.", lw=lw)
ax2.plot(xdata, tot10 / 1e6, label=r"I$_p$(10$^{25}$)", linestyle="-.", lw=lw)
ax2.plot(xdata, re2 / 1e6, 'tab:blue', label=r"I$_{RE}$(2$\cdot 10^{24}$)", linestyle="--", lw=lw)
ax2.plot(xdata, re6 / 1e6, 'tab:orange', label=r"I$_{RE}$(6$\cdot 10^{24}$)", linestyle="--", lw=lw)
ax2.plot(xdata, re10 / 1e6, 'tab:green', label=r"I$_{RE}$(10$^{25}$)", linestyle="--", lw=lw)
ax2.legend(facecolor="white",loc=3)
ax2.tick_params(axis='y')
ax2.set_ylim([-0.05, None])
ax1.set_xlim([0, 31])
ax2.set_xlim([0, 31])