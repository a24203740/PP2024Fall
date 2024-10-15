#include <iostream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <array>
#include <utility>
#include <random>
#include <memory>

#include <pthread.h>
#include <time.h>
#include <stdlib.h>

#include <experimental/simd>

using ll = long long;
namespace stdx {
  using namespace std::experimental;
  using namespace std::experimental::__proposed;
}
using V_FLOAT = stdx::fixed_size_simd<float, 8>;
// rebind_simd_t can specify the type of the element in the new vector, and the number of elements would be the same as the original vector
// in this case, V_INT is a vector of uint32_t with the same number of elements as V_FLOAT
// ref: https://en.cppreference.com/w/cpp/experimental/simd/rebind_simd
using V_INT = stdx::rebind_simd_t<uint32_t, V_FLOAT>;
using V_MASK = stdx::fixed_size_simd_mask<float, 8>;

const size_t V_SIZE = V_FLOAT::size();
const V_FLOAT vOneFloat = V_FLOAT(1.0f);
const V_INT vMaxInt = V_INT(std::numeric_limits<uint32_t>::max());

// we need to used static_simd_cast because not all int value can represent as float
// ref: https://en.cppreference.com/w/cpp/experimental/simd/simd_cast 
const V_FLOAT vMaxDouble = stdx::static_simd_cast<V_FLOAT>(vMaxInt);


static inline V_INT xorshift32(V_INT x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

static inline V_FLOAT uniformDistribution(V_INT &state) {
    return stdx::static_simd_cast<V_FLOAT>(state) / vMaxDouble;
}

struct ThreadData {
    ll n;
    V_INT seed;
    ll inside;
};

void* MonteCarloPi(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    ll n = td->n;
    V_INT state = td->seed;
    ll inside = 0;
    ll n_rem = n % V_SIZE;
    ll n_vec = n - n_rem;
    for (ll i = 0; i < n_vec; i+=V_SIZE) {
        state = xorshift32(state);
        V_FLOAT x = uniformDistribution(state);
        state = xorshift32(state);
        V_FLOAT y = uniformDistribution(state);
        V_FLOAT r = x * x + y * y;
        // psuedo: for(ri in r) if(ri <= 1) inside++;
        inside += stdx::popcount(r <= vOneFloat); 
    }
    if(n_rem > 0)
    {
        state = xorshift32(state);
        V_FLOAT x = uniformDistribution(state);
        state = xorshift32(state);
        V_FLOAT y = uniformDistribution(state);
        V_FLOAT r = x * x + y * y;
        for(ll i = 0; i < n_rem; i++)
        {
            if(r[i] <= 1)
            {
                inside++;
            }
        }
    }
    td->inside = inside;
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
    
    std::vector<V_INT> seeds(tn);
    std::vector<pthread_t> threads(tn);
    std::vector<ThreadData> threadDataes(tn);

    for(unsigned int i = 0; i < tn; i++)
    {
        seeds[i] = V_INT([&rd](int i){return rd();});
    }

    for(unsigned int i = 0, rem = n % tn; i < tn; i++)
    {
        threadDataes[i] = {n / tn, seeds[i], 0};
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
    ll realN = 0;
    for(unsigned int i = 0; i < tn; i++)
    {
        inside += threadDataes[i].inside;
        realN += threadDataes[i].n;
    }
    double pi = 4.0 * inside / realN;
    std::cout << std::fixed << std::setprecision(8) << pi << std::endl;
    return 0;   
}