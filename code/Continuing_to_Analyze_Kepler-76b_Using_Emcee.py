# 2018 Oct 9 - Continuing to analyze

import dill

in_filename = 'Analyzing_Kepler-76b_Using_Emcee.pkl'
dill.load_session(in_filename)

# Recover last position
pos = sampler.chain[:,-1,:]

nsteps = 5000
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    if (i+1) % 50 == 0:
        print("{0:5.1%}".format(float(i) / nsteps))

out_filename = 'Continuing_to_Analyze_Kepler-76b_Using_Emcee.pkl'
dill.dump_session(out_filename)
print(np.mean(sampler.chain[:, :, 0]), np.std(sampler.chain[:, :, 0]))
