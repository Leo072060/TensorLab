#ifndef UNCOPYABLE_HPP
#define UNCOPYABLE_HPP

class Uncopyable
{
  public:
    Uncopyable() {}
    ~Uncopyable() {}

  private:
    Uncopyable(const Uncopyable &)            = delete;
    Uncopyable(Uncopyable &&)                 = delete;
    Uncopyable &operator=(const Uncopyable &) = delete;
    Uncopyable &operator=(Uncopyable &&)      = delete;
};

#endif // UNCOPYABLE_HPP