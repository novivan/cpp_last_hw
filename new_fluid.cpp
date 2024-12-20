#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <cstring>
#include <chrono>
#include <random>
#include <array>
#include <cassert>
#include <tuple>
#include <algorithm>
#include <ranges>
//#include <recursive_mutex> // Добавьте этот include

using namespace std;

constexpr size_t N = 36, M = 84;
constexpr size_t T = 1'000'000;
constexpr std::array<pair<int, int>, 4> deltas{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

char field[N][M + 1] = {
    "####################################################################################",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                       .........                                  #",
    "#..............#            #           .........                                  #",
    "#..............#            #           .........                                  #",
    "#..............#            #           .........                                  #",
    "#..............#            #                                                      #",
    "#..............#            #                                                      #",
    "#..............#            #                                                      #",
    "#..............#            #                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............#                                                      #",
    "#..............#............################                     #                 #",
    "#...........................#....................................#                 #",
    "#...........................#....................................#                 #",
    "#...........................#....................................#                 #",
    "##################################################################                 #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "#                                                                                  #",
    "####################################################################################",
};

struct Fixed {
    constexpr Fixed(int v): v(v << 16) {}
    constexpr Fixed(float f): v(f * (1 << 16)) {}
    constexpr Fixed(double f): v(f * (1 << 16)) {}
    constexpr Fixed(): v(0) {}

    static constexpr Fixed from_raw(int32_t x) {
        Fixed ret;
        ret.v = x;
        return ret;
    } 

    int32_t v;

    auto operator<=>(const Fixed&) const = default;
    bool operator==(const Fixed&) const = default;
};

static constexpr Fixed inf = Fixed::from_raw(std::numeric_limits<int32_t>::max());
static constexpr Fixed eps = Fixed::from_raw(deltas.size());

Fixed operator+(Fixed a, Fixed b) {
    return Fixed::from_raw(a.v + b.v);
}

Fixed operator-(Fixed a, Fixed b) {
    return Fixed::from_raw(a.v - b.v);
}

Fixed operator*(Fixed a, Fixed b) {
    return Fixed::from_raw(((int64_t) a.v * b.v) >> 16);
}

Fixed operator/(Fixed a, Fixed b) {
    return Fixed::from_raw(((int64_t) a.v << 16) / b.v);
}

Fixed &operator+=(Fixed &a, Fixed b) {
    return a = a + b;
}

Fixed &operator-=(Fixed &a, Fixed b) {
    return a = a - b;
}

Fixed &operator*=(Fixed &a, Fixed b) {
    return a = a * b;
}

Fixed &operator/=(Fixed &a, Fixed b) {
    return a = a / b;
}

Fixed operator-(Fixed x) {
    return Fixed::from_raw(-x.v);
}

Fixed abs(Fixed x) {
    if (x.v < 0) {
        x.v = -x.v;
    }
    return x;
}

ostream &operator<<(ostream &out, Fixed x) {
    return out << x.v / (double) (1 << 16);
}

Fixed rho[256];

Fixed p[N][M]{}, old_p[N][M];

// Глобальные мьютексы для защиты общих ресурсов
std::mutex velocity_mutex;
std::mutex field_mutex;
std::mutex p_mutex;
std::recursive_mutex last_use_mutex;  // Изменено с std::mutex на std::recursive_mutex

// Модифицируем VectorField для безопасного доступа
struct VectorField {
    array<Fixed, deltas.size()> v[N][M];
    mutable std::mutex mtx;
    
    VectorField() = default;
    
    // Добавляем метод для очистки вместо присваивания
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        for(size_t i = 0; i < N; i++) {
            for(size_t j = 0; j < M; j++) {
                for(size_t k = 0; k < deltas.size(); k++) {
                    v[i][j][k] = Fixed(0);
                }
            }
        }
    }

    Fixed get_safe(int x, int y, int dx, int dy) const {
        std::lock_guard<std::mutex> lock(mtx);
        size_t i = ranges::find(deltas, pair(dx, dy)) - deltas.begin();
        assert(i < deltas.size());
        return v[x][y][i];
    }
    
    void set_safe(int x, int y, int dx, int dy, Fixed value) {
        std::lock_guard<std::mutex> lock(mtx);
        size_t i = ranges::find(deltas, pair(dx, dy)) - deltas.begin();
        assert(i < deltas.size());
        v[x][y][i] = value;
    }

    Fixed &add(int x, int y, int dx, int dy, Fixed dv) {
        std::lock_guard<std::mutex> lock(mtx);
        size_t i = ranges::find(deltas, pair(dx, dy)) - deltas.begin();
        assert(i < deltas.size());
        return v[x][y][i] += dv;
    }

    Fixed &get(int x, int y, int dx, int dy) {
        std::lock_guard<std::mutex> lock(mtx);
        size_t i = ranges::find(deltas, pair(dx, dy)) - deltas.begin();
        assert(i < deltas.size());
        return v[x][y][i];
    }
};

VectorField velocity{}, velocity_flow{};
int last_use[N][M]{};
int UT = 0;

mt19937 rnd(1337);

tuple<Fixed, bool, pair<int, int>> propagate_flow(int x, int y, Fixed lim) {
    last_use[x][y] = UT - 1;
    Fixed ret = 0;
    for (auto [dx, dy] : deltas) {
        int nx = x + dx, ny = y + dy;

        // Проверка границ массива
        if (nx < 0 || nx >= N || ny < 0 || ny >= M) {
            continue;
        }

        if (field[nx][ny] != '#' && last_use[nx][ny] < UT) {
            auto cap = velocity.get(x, y, dx, dy);
            auto flow = velocity_flow.get(x, y, dx, dy);
            if (flow == cap) {
                continue;
            }
            auto vp = min(lim, cap - flow);
            if (last_use[nx][ny] == UT - 1) {
                velocity_flow.add(x, y, dx, dy, vp);
                last_use[x][y] = UT;
                return {vp, 1, {nx, ny}};
            }
            auto [t, prop, end] = propagate_flow(nx, ny, vp);
            ret += t;
            if (prop) {
                velocity_flow.add(x, y, dx, dy, t);
                last_use[x][y] = UT;
                return {t, prop && end != pair(x, y), end};
            }
        }
    }
    last_use[x][y] = UT;
    return {ret, 0, {0, 0}};
}

Fixed random01() {
    return Fixed::from_raw((rnd() & ((1 << 16) - 1)));
}

void propagate_stop(int x, int y, bool force = false) {
    std::lock_guard<std::recursive_mutex> lock(last_use_mutex);
    if (!force) {
        bool stop = true;
        for (auto [dx, dy] : deltas) {
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) > 0) {
                stop = false;
                break;
            }
        }
        if (!stop) {
            return;
        }
    }
    last_use[x][y] = UT;
    for (auto [dx, dy] : deltas) {
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT || velocity.get(x, y, dx, dy) > 0) {
            continue;
        }
        propagate_stop(nx, ny);
    }
}

Fixed move_prob(int x, int y) {
    Fixed sum = 0;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
            continue;
        }
        auto v = velocity.get(x, y, dx, dy);
        if (v < 0) {
            continue;
        }
        sum += v;
    }
    return sum;
}

struct ParticleParams {
    char type;
    Fixed cur_p;
    array<Fixed, deltas.size()> v;

    void swap_with(int x, int y) {
        swap(field[x][y], type);
        swap(p[x][y], cur_p);
        swap(velocity.v[x][y], v);
    }
};

bool propagate_move(int x, int y, bool is_first) {
    {
        std::lock_guard<std::recursive_mutex> lock(last_use_mutex);
        last_use[x][y] = UT - is_first;
    }
    
    bool ret = false;
    int nx = -1, ny = -1;
    do {
        std::array<Fixed, deltas.size()> tres;
        Fixed sum = 0;
        for (size_t i = 0; i < deltas.size(); ++i) {
            auto [dx, dy] = deltas[i];
            int nx_new = x + dx, ny_new = y + dy;

            // Проверка границ массива
            if (nx_new < 0 || nx_new >= N || ny_new < 0 || ny_new >= M) {
                tres[i] = sum;
                continue;
            }

            if (field[nx_new][ny_new] == '#' || last_use[nx_new][ny_new] == UT) {
                tres[i] = sum;
                continue;
            }
            auto v = velocity.get(x, y, dx, dy);
            if (v < 0) {
                tres[i] = sum;
                continue;
            }
            sum += v;
            tres[i] = sum;
        }

        if (sum == 0) {
            break;
        }

        Fixed p_val = random01() * sum;
        size_t d = std::ranges::upper_bound(tres, p_val) - tres.begin();

        auto [dx, dy] = deltas[d];
        nx = x + dx;
        ny = y + dy;

        // Дополнительная проверка границ
        if (nx < 0 || nx >= N || ny < 0 || ny >= M) {
            continue;
        }

        bool valid_move;
        {
            std::lock_guard<std::recursive_mutex> lock(last_use_mutex);
            valid_move = velocity.get_safe(x, y, dx, dy) > 0 && 
                        field[nx][ny] != '#' && 
                        last_use[nx][ny] < UT;
        }
        
        if (!valid_move) {
            continue;
        }

        {
            std::lock_guard<std::recursive_mutex> lock(last_use_mutex);
            ret = (last_use[nx][ny] == UT - 1 || propagate_move(nx, ny, false));
        }
    } while (!ret);

    {
        std::lock_guard<std::recursive_mutex> lock(last_use_mutex);
        last_use[x][y] = UT;
        
        for (auto [dx, dy] : deltas) {
            int nx_new = x + dx, ny_new = y + dy;

            // Проверка границ массива
            if (nx_new < 0 || nx_new >= N || ny_new < 0 || ny_new >= M) {
                continue;
            }

            if (field[nx_new][ny_new] != '#' && last_use[nx_new][ny_new] < UT - 1 && 
                velocity.get_safe(x, y, dx, dy) < 0) {
                propagate_stop(nx_new, ny_new);
            }
        }
        
        if (ret && !is_first) {
            ParticleParams pp{};
            {
                std::lock_guard<std::mutex> lock(field_mutex);
                pp.swap_with(x, y);
                pp.swap_with(nx, ny);
                pp.swap_with(x, y);
            }
        }
    }
    
    return ret;
}

int dirs[N][M]{};

// Пул потоков
class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();
    std::future<void> enqueue(std::function<void()> task);

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for(size_t i = 0; i < num_threads; ++i)
        workers.emplace_back([this]{
            while(true){
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                    if(this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
}

std::future<void> ThreadPool::enqueue(std::function<void()> task){
    auto packaged_task = std::make_shared<std::packaged_task<void()>>(task);
    std::future<void> res = packaged_task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace([packaged_task](){ (*packaged_task)(); });
    }
    condition.notify_one();
    return res;
}

ThreadPool::~ThreadPool(){
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(auto &worker: workers)
        worker.join();
}

std::atomic<bool> prop{false};
std::mutex state_mutex;
std::condition_variable state_cv;

void recalculate_p(size_t start_x, size_t end_x) {
    for (size_t x = start_x; x < end_x; ++x) {
        for (size_t y = 0; y < M; ++y) {
            if (field[x][y] == '#')
                continue;
            for (auto [dx, dy] : deltas) {
                auto old_v = velocity.get_safe(x, y, dx, dy);
                auto new_v = velocity_flow.get_safe(x, y, dx, dy);
                if (old_v > 0) {
                    // Безопасная корректировка значения
                    new_v = min(new_v, old_v);
                    velocity.set_safe(x, y, dx, dy, new_v);
                    
                    std::lock_guard<std::mutex> lock(p_mutex);
                    auto force = (old_v - new_v) * rho[(int) field[x][y]];
                    if (field[x][y] == '.')
                        force *= 0.8;
                    if (field[x + dx][y + dy] == '#') {
                        p[x][y] += force / dirs[x][y];
                    } else {
                        p[x + dx][y + dy] += force / dirs[x + dx][y + dy];
                    }
                }
            }
        }
    }
}

// Заменяем обычную булеву переменную на атомарную для безопасного доступа из разных потоков
std::atomic<bool> global_prop{false};

// Добавляем глобальную блокировку для отладки
std::mutex global_mutex;
std::atomic<int> active_threads{0};

int main(int argc, char* argv[]) {
    size_t num_threads = std::thread::hardware_concurrency();
    if(argc > 1) {
        num_threads = std::stoi(argv[1]);
    }
    cout << "Running with " << num_threads << " threads" << endl;
    
    ThreadPool pool(num_threads);

    rho[' '] = 0.01;
    rho['.'] = 1000;
    Fixed g = 0.1;

    for (size_t x = 0; x < N; ++x) {
        for (size_t y = 0; y < M; ++y) {
            if (field[x][y] == '#')
                continue;
            for (auto [dx, dy] : deltas) {
                dirs[x][y] += (field[x + dx][y + dy] != '#');
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < T; ++i) {
        if (i % 100 == 0) {
            cout << "Iteration " << i << "/" << T << endl;
        }

        std::vector<std::future<void>> futures;
        
        // Apply external forces in single thread for safety
        {
            std::lock_guard<std::mutex> lock(global_mutex);
            for (size_t x = 0; x < N; ++x) {
                for (size_t y = 0; y < M; ++y) {
                    if (field[x][y] == '#') continue;
                    // Проверка границ перед доступом
                    if (x + 1 < N && field[x + 1][y] != '#')
                        velocity.add(x, y, 1, 0, g);
                }
            }
        }

        // Copy old_p to ensure consistency
        memcpy(old_p, p, sizeof(p));

        // Apply forces from p in chunks
        const size_t chunk_size = N / num_threads;
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? N : (t + 1) * chunk_size;
            
            futures.push_back(pool.enqueue([&, start, end]() {
                // Избегаем захвата global_mutex здесь, если возможно
                for (size_t x = start; x < end; ++x) {
                    for (size_t y = 0; y < M; ++y) {
                        if (field[x][y] == '#')
                            continue;
                        for (auto [dx, dy] : deltas) {
                            int nx = x + dx, ny = y + dy;
                            // Проверка границ массива
                            if (nx < 0 || nx >= N || ny < 0 || ny >= M)
                                continue;
                            if (field[nx][ny] != '#' && old_p[nx][ny] < old_p[x][y]) {
                                auto delta_p = old_p[x][y] - old_p[nx][ny];
                                auto force = delta_p;
                                auto &contr = velocity.get(nx, ny, -dx, -dy);
                                if (contr * rho[(int) field[nx][ny]] >= force) {
                                    contr -= force / rho[(int) field[nx][ny]];
                                    continue;
                                }
                                force -= contr * rho[(int) field[nx][ny]];
                                contr = 0;
                                velocity.add(x, y, dx, dy, force / rho[(int) field[x][y]]);
                                p[x][y] -= force / dirs[x][y];
                            }
                        }
                    }
                }
            }));
        }

        for(auto &fut : futures) {
            fut.get();
        }
        futures.clear();

        // Make flow from velocities
        velocity_flow.clear();
        std::atomic<bool> flow_continues{false}; // Уже атомарная переменная
        do {
            flow_continues.store(false, std::memory_order_relaxed); // Сброс флага
            UT += 2;

            for (size_t x = 0; x < N; ++x) {
                futures.push_back(pool.enqueue([&, x]() {
                    // Избегаем захвата global_mutex внутри потоков при обработке flow
                    for (size_t y = 0; y < M; ++y) {
                        if (field[x][y] != '#' && last_use[x][y] != UT) {
                            auto [t, local_prop, _] = propagate_flow(x, y, 1);
                            if (t > 0) {
                                flow_continues.store(true, std::memory_order_relaxed); // Установка флага
                            }
                        }
                    }
                }));
            }

            for(auto &fut : futures) {
                fut.get();
            }
            futures.clear();
        } while (flow_continues.load(std::memory_order_relaxed)); // Проверка флага

        // Process movement sequentially for now
        {
            std::lock_guard<std::mutex> lock(global_mutex);
            UT += 2;
            bool moved = false;
            for (size_t x = 0; x < N; ++x) {
                for (size_t y = 0; y < M; ++y) {
                    if (field[x][y] != '#' && last_use[x][y] != UT) {
                        if (random01() < move_prob(x, y)) {
                            moved = true;
                            propagate_move(x, y, true);
                        } else {
                            propagate_stop(x, y, true);
                        }
                    }
                }
            }

            if (moved) {
                cout << "Tick " << i << ":\n";
                for (size_t x = 0; x < N; ++x) {
                    cout << field[x] << "\n";
                }
                cout << flush;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout << "Время выполнения: " << diff.count() << " секунд\n";

    
    return 0;
}
