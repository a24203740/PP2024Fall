CC := g++
CXXFLAGS := -O3 -std=c++17 -Wall
CXXLIBS := -lm -lpthread

APP_NAME := pi.out

default: $(APP_NAME)

PHONY: clean

OBJS := pi.o

$(APP_NAME): $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $(OBJS) $(CXXLIBS)

%.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) *.o $(APP_NAME)
