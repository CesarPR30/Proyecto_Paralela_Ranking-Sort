# Compilacion en Linux / cluster (OpenMPI o MPICH)
#   make            compila todo en bin/
#   make clean
CXX      = mpic++
CXXFLAGS = -O3 -std=c++17 -Wall
BIN      = bin

all: $(BIN)/rs_sec $(BIN)/rs_beta1 $(BIN)/rs_beta2 $(BIN)/rs_beta3 $(BIN)/qs_par $(BIN)/rs_hibrido

$(BIN):
	mkdir -p $(BIN)

$(BIN)/rs_sec: ranking_sort_sec.cpp | $(BIN)
	g++ $(CXXFLAGS) -o $@ $<

$(BIN)/rs_beta1: ranking_sort_beta1.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/rs_beta2: ranking_sort_beta2.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/rs_beta3: ranking_sort_beta3.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/qs_par: quicksort_paralelo.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/rs_hibrido: ranking_sort_hibrido.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<

clean:
	rm -rf $(BIN)

.PHONY: all clean
