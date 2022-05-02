import numpy as np

def gaussian_generator(mean, variance):

    x = np.random.normal(mean, variance)

    return x

def bayes_probability(a, xs):

    prob = (1/(a * np.sqrt(2 * np.pi)))**(len(xs))

    for x in xs:
        prob = prob * np.exp(-x**2/(2*a**2))

    return prob

def bayes_distribution(variance, a_vals, num_samples):

    xs = []

    for _ in range(num_samples):
        xs.append(gaussian_generator(0,variance))

    probs = []
    for i in range(len(a_vals)):
        probs.append(bayes_probability(a_vals[i], xs)/len(a_vals))

    plt.hist(xs)
    plt.show()

    return probs

def generating_gaussian(a_prime, a_t, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(a_prime - a_t)**2/(2*sigma**2))

def metropolis_hastings(a_vals, max_a, posterior_dist):

    sigma = 0.2
    a_0 = np.random.choice(a_vals)
    t_max = 10000

    a_path = [a_0]

    for _ in np.arange(0,t_max):
        
        a_prime = gaussian_generator(a_path[-1], sigma)
        while a_prime <= 0 or a_prime > max_a:
            a_prime = gaussian_generator(a_path[-1], sigma)
        
        P_xprime = posterior_dist[int(np.floor(a_prime*len(a_vals)/max_a))]
        P_xt = posterior_dist[int(np.floor(a_path[-1]*len(a_vals)/max_a))]
        g_xt_xprime = generating_gaussian(a_path[-1],a_prime,sigma)
        g_xprime_xt = generating_gaussian(a_prime,a_path[-1],sigma)

        accept_prob = np.min([1, P_xprime / P_xt * g_xt_xprime / g_xprime_xt])
        
        if np.random.uniform() > accept_prob:
            a_path.append(a_path[-1])
        else:
            a_path.append(a_prime)

    return a_path

num_samples = 100
actual_variance = 3
num_as = 20000
max_a = 10

a_vals = np.linspace(max_a/num_as,max_a,num_as)

posterior_distribution = bayes_distribution(actual_variance, a_vals, num_samples)

plt.plot(a_vals, posterior_distribution)
plt.show()

path = metropolis_hastings(a_vals, max_a, posterior_distribution)

plt.plot(path)
plt.show()

plt.hist(path)
plt.axvline(x=actual_variance, color='r')
plt.show()