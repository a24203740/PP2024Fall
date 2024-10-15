#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>
#include <random>
#include <memory>

#include <pthread.h>

using ll = long long;

struct ThreadData {
    ll n;
    std::seed_seq* seeds;
    ll inside;
};

void* MonteCarloPi(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    ll n = td->n;
    std::mt19937 gen(*td->seeds);
    std::uniform_real_distribution<double> dis(0, 1);
    if(n <= UINT32_MAX)
    {
        uint inside = 0;
        uint n32 = n;
        for (uint i = 0; i < n32; i++) {
            double x = dis(gen);
            double y = dis(gen);
            if (x * x + y * y <= 1) {
                inside++;
            }
        }
        td->inside = inside;
    }
    else
    {
        ll inside = 0;
        for (ll i = 0; i < n; i++) {
            double x = dis(gen);
            double y = dis(gen);
            if (x * x + y * y <= 1) {
                inside++;
            }
        }
        td->inside = inside;
    }
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    if(argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <number of thread> <number of tosses>" << std::endl;
        return 1;
    }
    unsigned int tn = std::stoul(argv[1]);
    unsigned int n = std::stoul(argv[2]);
    
    std::random_device rd;
    
    std::vector<std::unique_ptr<std::seed_seq>> seeds(tn);
    std::vector<pthread_t> threads(tn);
    std::vector<ThreadData> threadDataes(tn);
    
    for (unsigned int i = 0; i < tn; i++) {
        std::vector<unsigned int> seedVal{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        // std::vector<unsigned int> seedVal{1,2,3,4,5,6,7,8};
        seeds[i] = std::make_unique<std::seed_seq>(seedVal.begin(), seedVal.end());
    }

    for(unsigned int i = 0, rem = n % tn; i < tn; i++)
    {
        threadDataes[i] = {n / tn, seeds[i].get(), 0};
        if(rem > 0)
        {
            threadDataes[i].n++;
            rem--;
        }
    }

    for(unsigned int i = 0; i < tn; i++)
    {
        pthread_create(&threads[i], NULL, MonteCarloPi, (void*)(threadDataes.data() + i));
    }

    for(unsigned int i = 0; i < tn; i++)
    {
        pthread_join(threads[i], NULL);
    }

    ll inside = 0;
    for(unsigned int i = 0; i < tn; i++)
    {
        inside += threadDataes[i].inside;
    }
    double pi = 4.0 * inside / n;
    std::cout << std::fixed << std::setprecision(8) << pi << std::endl;
    return 0;   
}