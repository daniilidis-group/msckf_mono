#pragma once

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include <iostream>
#include <fstream>
#include <type_traits>
#include <string>
#include <tuple>
#include <memory>
#include <utility>
#include <algorithm>

namespace synchronizer
{
  template <typename T> struct has_get_data {
    template <typename U> static  std::true_type f(decltype(&U::get_data));
    template <typename U> static std::false_type f(...);

    typedef decltype(f<T>(0)) type;
  };

  template <typename T> struct has_get_time {
    template <typename U> static  std::true_type f(decltype(&U::get_time));
    template <typename U> static std::false_type f(...);

    typedef decltype(f<T>(0)) type;
  };

  template <typename T> struct has_next {
    template <typename U> static  std::true_type f(decltype(&U::next));
    template <typename U> static std::false_type f(...);

    typedef decltype(f<T>(0)) type;
  };

  template <typename T> struct has_has_next {
    template <typename U> static  std::true_type f(decltype(&U::has_next));
    template <typename U> static std::false_type f(...);

    typedef decltype(f<T>(0)) type;
  };

  template<class...> struct conjunction : std::true_type { };
  template<class B1> struct conjunction<B1> : B1 { };
  template<class B1, class... Bn>
    struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

  template <typename T, typename F, size_t... Is>
    constexpr auto apply_impl(T& t, F& f, std::index_sequence<Is...>) {;
      return f(std::get<Is>(t)...);
    }

  template <typename T, typename F>
  constexpr auto apply(T& t, F&& f) {
    return apply_impl(t, f, std::make_index_sequence<std::tuple_size<T>{}>{});
  }

  template <typename T, typename... Ts>
  struct first_template_arg {
    typedef T type;
  };

  /*
   * This class will temporally synchronize feeds from multiple sensors.
   *
   * A sensor must contain the following functions
   *  - has_next
   *  - next
   *  - get_data
   *  - get_time
   *
   *  Because the implementation probes for existence of the functions at compile time,
   *  they don't need to inherit from any particular class and can be written completely separately.
   */

  template<typename... Ts>
  class Synchronizer
  {
    public:
      template<typename T>
        static bool conditional_next(T& s, bool b)
        {
          if(b){
            return s.next();
          }
          return true;
        }

      template<typename T>
        static boost::optional<std::result_of_t<decltype(&T::get_data)(T)>> conditional_get_data(T& s, bool b)
        {
          if(b){
            return boost::optional<std::result_of_t<decltype(&T::get_data)(T)>>(s.get_data());
          }else{
            return boost::none;
          }
        }

      typedef typename std::tuple<std::shared_ptr<Ts> ...> SensorPack;

      typedef typename first_template_arg<Ts...>::type first_Ts;
      typedef std::result_of_t<decltype(&first_Ts::get_time)(first_Ts)> TimeType;
      typedef typename std::tuple<boost::optional<std::result_of_t<decltype(&Ts::get_data)(Ts)>> ...> SensorDataPack;

      Synchronizer(std::shared_ptr<Ts>... Sensors) : sensors_(std::make_tuple(Sensors...))
    {
      static_assert(conjunction<typename has_next<Ts>::type ...>::value,
          "All sensors must implement next.");
      static_assert(conjunction<typename has_has_next<Ts>::type ...>::value,
          "All sensors must implement has_next.");
      static_assert(conjunction<typename has_get_time<Ts>::type ...>::value,
          "All sensors must implement get_time.");
      static_assert(conjunction<typename has_get_data<Ts>::type ...>::value,
          "All sensors must implement get_data.");

      static_assert(std::tuple_size<SensorPack>::value > 0, "Must have at least one sensor.");
    };

      SensorDataPack get_data()
      {
        auto master_time = apply(sensors_, [](auto...s){return std::min({s->get_time()...});});
        SensorDataPack sdp = apply(sensors_, [&master_time](auto...s)
            {
              return std::make_tuple(Synchronizer::conditional_get_data(*s, s->get_time()<=master_time) ... );
            });
        return sdp;
      }

      TimeType get_time()
      {
        auto master_time = apply(sensors_, [](auto...s){return std::min({s->get_time()...});});
        return TimeType(master_time);
      };

      bool next()
      {
        if(!has_next())
          return false;

        auto master_time = apply(sensors_, [](auto...s){return std::min({s->get_time()...});});
        apply(sensors_, [&master_time](auto...s)
            {
              return std::make_tuple(Synchronizer::conditional_next(*s, s->get_time()<=master_time) ... );
            });

        return true;
      }

      bool has_next()
      {
        return apply(sensors_, [](auto...s){return std::min({s->has_next()...});});
      }

    private:
      SensorPack sensors_;
  };

  template<typename... Ts>
    auto make_synchronizer(std::shared_ptr<Ts>... args ){
      return Synchronizer<Ts...>(args ...);
    }
}
