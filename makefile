NAME = TBD

CFLAGS = -W -Wall -Wextra

SRC = $(wildcard ./src/*.c) \
		$(wildcard ./src/func/*.c) \
			$(wildcard ./src/libs/*.c)

OBJ = $(SRC:.c=.o)

all: build clean

build: $(OBJ)
	@if not exist "./out" mkdir "./out"
	@gcc $(CFLAGS) $(OBJ) -o ./out/$(NAME)

clean:
	@if exist ./src/*.o del /Q src\*.o
	@if exist ./src/func/*.o del /Q src\func\*.o
	@if exist ./src/libs/*.o del /Q src\libs\*.o