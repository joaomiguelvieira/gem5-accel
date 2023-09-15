CC=g++
CCF=-g
LDF=-static
SRC_O0=main.cpp
SRC_O3=baseline.cpp
SRC=$(SRC_O0) $(SRC_O3)
OBJ=$(SRC:.cpp=.o)
EXE=bench

all: $(SRC) $(EXE)

$(EXE): $(OBJ)
	$(CC) $(CCF) $(OBJ) -o $@ $(LDF)

$(SRC_O0:.cpp=.o): $(SRC_O0)
	$(CC) $(CCF) -O0 -c $< -o $@ $(LDF)

$(SRC_O3:.cpp=.o): $(SRC_O3)
	$(CC) $(CCF) -O3 -c $< -o $@ $(LDF)

clean:
	rm -f $(OBJ) $(EXE)
