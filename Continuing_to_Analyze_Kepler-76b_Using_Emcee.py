# 2018 Oct 9 - Just copying over the useful bits

filename = 'Continuing_to_Analyze_Kepler-76b_Using_Emcee.pkl'
dill.load_session(filename)

# Recover last position
pos = sampler.chain[:,-1,:]

nsteps = 5000
for i, result in enumerate(sampler.run_mcmc(pos, iterations=nsteps)):
    if (i+1) % 50 == 0:
        print("{0:5.1%}".format(float(i) / nsteps))

dill.dump_session(filename)
print(np.mean(sampler.chain[:, :, 0]), np.std(sampler.chain[:, :, 0]))

