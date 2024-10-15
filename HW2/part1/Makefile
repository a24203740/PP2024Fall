CC := g++
# CXXFLAGS := -O3 -std=c++20 -Wall -Ofast -Winline -fgcse -fgcse-lm -fipa-sra -ftree-pre -ftree-vrp -fpeephole2 -ffast-math -fsched-spec -unroll-loops -falign-jumps -falign-loops -falign-labels -fdevirtualize -fcaller-saves -fcrossjumping -fthread-jumps -funroll-loops -fwhole-program -freorder-blocks -fschedule-insns -finline-functions -ftree-tail-merge -fschedule-insns2 -fstrict-aliasing -fstrict-overflow -falign-functions -fcse-skip-blocks -fcse-follow-jumps -fsched-interblock -fpartial-inlining -Wno-stack-protector -freorder-functions -findirect-inlining -frerun-cse-after-loop -finline-small-functions -ftree-switch-conversion -foptimize-sibling-calls -fexpensive-optimizations -funsafe-loop-optimizations -finline-functions-called-once -fdelete-null-pointer-checks
CXXFLAGS := -O3 -std=c++20 -Wall -Ofast
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

#pragma G++ optimize(1)
#pragma G++ optimize(2)
#pragma G++ optimize(3)
#pragma G++ optimize("Ofast")
#pragma G++ optimize("inline")
#pragma G++ optimize("-fgcse")
#pragma G++ optimize("-fgcse-lm")
#pragma G++ optimize("-fipa-sra")
#pragma G++ optimize("-ftree-pre")
#pragma G++ optimize("-ftree-vrp")
#pragma G++ optimize("-fpeephole2")
#pragma G++ optimize("-ffast-math")
#pragma G++ optimize("-fsched-spec")
#pragma G++ optimize("unroll-loops")
#pragma G++ optimize("-falign-jumps")
#pragma G++ optimize("-falign-loops")
#pragma G++ optimize("-falign-labels")
#pragma G++ optimize("-fdevirtualize")
#pragma G++ optimize("-fcaller-saves")
#pragma G++ optimize("-fcrossjumping")
#pragma G++ optimize("-fthread-jumps")
#pragma G++ optimize("-funroll-loops")
#pragma G++ optimize("-fwhole-program")
#pragma G++ optimize("-freorder-blocks")
#pragma G++ optimize("-fschedule-insns")
#pragma G++ optimize("inline-functions")
#pragma G++ optimize("-ftree-tail-merge")
#pragma G++ optimize("-fschedule-insns2")
#pragma G++ optimize("-fstrict-aliasing")
#pragma G++ optimize("-fstrict-overflow")
#pragma G++ optimize("-falign-functions")
#pragma G++ optimize("-fcse-skip-blocks")
#pragma G++ optimize("-fcse-follow-jumps")
#pragma G++ optimize("-fsched-interblock")
#pragma G++ optimize("-fpartial-inlining")
#pragma G++ optimize("no-stack-protector")
#pragma G++ optimize("-freorder-functions")
#pragma G++ optimize("-findirect-inlining")
#pragma G++ optimize("-frerun-cse-after-loop")
#pragma G++ optimize("inline-small-functions")
#pragma G++ optimize("-finline-small-functions")
#pragma G++ optimize("-ftree-switch-conversion")
#pragma G++ optimize("-foptimize-sibling-calls")
#pragma G++ optimize("-fexpensive-optimizations")
#pragma G++ optimize("-funsafe-loop-optimizations")
#pragma G++ optimize("inline-functions-called-once")
#pragma G++ optimize("-fdelete-null-pointer-checks")