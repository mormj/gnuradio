#pragma once

#if 0
#include <moodycamel/blockingconcurrentqueue.h>

namespace gr {

/**
 * @brief Blocking Multi-producer Single-consumer Queue class
 *
 * @tparam T Data type of items in queue
 */
template <typename T>
class concurrent_queue
{
public:
    bool push(const T& msg)
    {
        q.enqueue(msg);
        return true;
    }

    // Non-blocking
    bool try_pop(T& msg) { return q.try_dequeue(msg); }
    bool pop(T& msg)
    {
        q.wait_dequeue(msg);
        return true;
    }
    void clear()
    {
        T msg;
        bool done = false;
        while (!done)
            done = !q.try_dequeue(msg);
    }
    size_t size_approx() { return q.size_approx(); }

private:
    moodycamel::BlockingConcurrentQueue<T> q;
};
} // namespace gr
#elif 0

#include <condition_variable>
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
using namespace std::chrono_literals;
namespace gr {

/**
 * @brief Blocking Multi-producer Single-consumer Queue class
 *
 * @tparam T Data type of items in queue
 */
template <typename T>
class concurrent_queue
{
public:
    bool push(const T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);
        // std::scoped_lock l(_mutex);
        _queue.push_back(msg);
        l.unlock();
        _cond.notify_one();

        return true;
    }

    // Non-blocking
    bool try_pop(T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);
        if (!_queue.empty()) {
            msg = _queue.front();
            _queue.pop_front();
            return true;
        }
        else {
            return false;
        }
    }
    bool pop(T& msg)
    {
#if 1
        std::unique_lock<std::mutex> l(_mutex);
        _cond.wait(l,
                   [this] { return !_queue.empty(); }); // TODO - replace with a waitfor
        msg = _queue.front();
        _queue.pop_front();
        return true;
#else
        std::unique_lock<std::mutex> l(_mutex);
        if (_cond.wait_for(l, 10us, [this] { return !_queue.empty(); })) {
            msg = _queue.front();
            _queue.pop_front();
            return true;
        }
        else { // timeout
            return false;
        }
#endif
    }
    void clear()
    {
        std::unique_lock<std::mutex> l(_mutex);
        _queue.clear();
    }

private:
    std::deque<T> _queue;
    std::mutex _mutex;
    std::condition_variable _cond;
};
} // namespace gr
#elif 0
// atomic flag
// described here:
// https://modernescpp.com/index.php/performancecomparison-of-condition-variables-and-atomics-in-c-20
#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
namespace gr {

/**
 * @brief Blocking Multi-producer Single-consumer Queue class
 *
 * @tparam T Data type of items in queue
 */
template <typename T>
class concurrent_queue
{
public:
    bool push(const T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);
        _queue.push_back(msg);
        l.unlock();
        _cond.test_and_set();
        _cond.notify_one();

        return true;
    }

    // Non-blocking
    bool try_pop(T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);
        if (!_queue.empty()) {
            msg = _queue.front();
            _queue.pop_front();
            return true;
        }
        else {
            return false;
        }
    }
    bool pop(T& msg)
    {
        _cond.wait(false);
        std::unique_lock<std::mutex> l(_mutex);
        msg = _queue.front();
        _queue.pop_front();
        l.unlock();
        _cond.clear();
        return true;
    }
    void clear()
    {
        std::unique_lock<std::mutex> l(_mutex);
        _queue.clear();
    }

private:
    std::deque<T> _queue;
    std::mutex _mutex;
    std::atomic_flag _cond{ true };
};
} // namespace gr
#else
#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
using namespace std::chrono_literals;
namespace gr {

/**
 * @brief Blocking Multi-producer Single-consumer Queue class
 *
 * @tparam T Data type of items in queue
 */
template <typename T>
class concurrent_queue
{
public:
    bool push(const T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);
        // std::scoped_lock l(_mutex);
        _queue.push_back(msg);
        l.unlock();
        _cond.test_and_set();
        _cond.notify_one();

        return true;
    }

    // Non-blocking
    bool try_pop(T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);
        if (!_queue.empty()) {
            msg = _queue.front();
            _queue.pop_front();
            return true;
        }
        else {
            return false;
        }
    }
    bool pop(T& msg)
    {
        while (true) {
            _cond.wait(false); // TODO - replace with a waitfor
            std::unique_lock<std::mutex> l(_mutex);
            if (_queue.empty()) {
                continue;
            }
            msg = _queue.front();
            _queue.pop_front();
            if (_queue.empty())
                _cond.clear();
            break;
        }
        return true;
    }
    void clear()
    {
        std::unique_lock<std::mutex> l(_mutex);
        _queue.clear();
    }

private:
    std::deque<T> _queue;
    std::mutex _mutex;
    std::atomic_flag _cond{};
};
} // namespace gr
#endif
