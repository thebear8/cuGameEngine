#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <Windows.h>
#include <list>

template<class retType, class fnType, class c, class... args>
struct binder
{
private:
	fnType fn;
	c* obj;

public:
	constexpr binder(fnType fn, c* obj)
	{
		this->fn = fn;
		this->obj = obj;
	}

	retType operator ()(args... a)
	{
		return (obj->*fn)(a...);
	}
};

template<class ret, class c, class... args>
constexpr auto bindMember(ret(c::* fn)(args...), c* obj)
{
	return binder<ret, ret(c::*)(args...), c, args...>(fn, obj);
}

template<class retType, class... args>
struct eventHandlerBase
{
public:
	virtual retType operator()(args... a) = 0;
};

template<class retType, class... args>
struct eventHandler : public eventHandlerBase<retType, args...>
{
private:
	retType(*handler)(args...);

public:
	constexpr eventHandler(retType(*handler)(args...))
	{
		this->handler = handler;
	}

	retType operator()(args... a)
	{
		return handler(a...);
	}
};

template<class retType, class fnType, class c, class... args>
struct boundEventHandler : public eventHandlerBase<retType, args...>
{
private:
	using boundHandlerType = binder<retType, fnType, c, args...>;
	boundHandlerType handler;

public:
	constexpr boundEventHandler(retType(c::* handlerFn)(args...), c* obj) : handler(handlerFn, obj)
	{
		
	}

	retType operator()(args... a)
	{
		return handler(a...);
	}
};

template<class retType, class... args>
constexpr auto createHandler(retType(*handler)(args...))
{
	return new eventHandler<retType, args...>(handler);
}

template<class retType, class c, class... args>
constexpr auto createBoundHandler(retType(c::* handler)(args...), c* obj)
{
	return new boundEventHandler<retType, retType(c::*)(args...), c, args...>(handler, obj);
}

template<class retType, class... args>
class eventListener
{
private:
	std::list<eventHandlerBase<retType, args...>*> subscribers;

public:
	void operator ()(args... a)
	{
		for (auto subscriber : subscribers)
		{
			(*subscriber)(a...);
		}
	}

	void operator +=(eventHandlerBase<retType, args...>* handler)
	{
		subscribers.push_back(handler);
	}

	void operator -=(eventHandlerBase<retType, args...>* handler)
	{
		subscribers.remove(handler);
	}
};