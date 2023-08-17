NAME = my_neuralnetwork

CFLAGS = -W -Wall -Wextra

LIBS = -l$(NAME) -lmy_matrix

SRC = $(wildcard ./src/*.c)

LIB_SRC = $(wildcard ./src/func/*.c)

LIB_OBJ = $(LIB_SRC:.c=.o)

OBJ = $(SRC:.c=.o)

all: archive build clean

archive: $(LIB_OBJ)
	ar -rcs C:\Users\ADMIN\Code\my_libs\lib$(NAME).a $(LIB_OBJ)

build: $(OBJ)
	@if not exist "./out" mkdir "./out"
	@gcc $(CFLAGS) $(OBJ) -LC:\Users\ADMIN\Code\my_libs $(LIBS) -o ./out/$(NAME)

clean:
	@if exist ./src/*.o del /Q src\*.o
	@if exist ./src/func/*.o del /Q src\func\*.o
