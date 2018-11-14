/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 
 */

#pragma once

#include <functional>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>

template <typename Parameter>
struct CallbackBuffer
{
  CallbackBuffer(char queueSize) { this->bufferSize = queueSize; }

  inline void addListener(const std::function<void(Parameter)>& listener) 
  {
    std::lock_guard<std::mutex> lock(mutex);// 上锁=========
    listeners.push_back(listener);
  }

  // Buffer size is checked seperately, see shrink().
  inline void addElement(const Parameter& e) 
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (buffer.size() > bufferSize) buffer.pop();
    buffer.emplace(e);// 优化之使用emplace 插入操作会涉及到两次构造
    // emplace 可保证 一次插入，一次构造================================================
  }

  // Pass all elements to all listeners (empty buffer)
  inline void callListeners()
  {
    // std::lock_guard<std::mutex> lock(mutex);
    while (!buffer.empty()) 
    {  // Fixme: Race condition
      for (auto& listener : listeners) 
      {
        listener(buffer.front());
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        buffer.pop();
      }
    }
  }

  inline void callListenersDirect(const Parameter& e)
  {
    for (auto& listener : listeners) listener(e);
  }

 private:
  std::vector<std::function<void(Parameter)> > listeners;
  std::queue<Parameter> buffer;
  std::mutex mutex;
  unsigned char bufferSize;
};

class Model;
typedef std::function<void(std::shared_ptr<Model>)> ModelListener;
