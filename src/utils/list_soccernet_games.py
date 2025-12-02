from SoccerNet.utils import getListGames
import os

def main():
    print("Consultando SoccerNet para obtener la lista de juegos...")
    try:
        # Intentamos usar getListGames con los parámetros que el usuario sugirió, adaptados
        # getListGames devuelve una lista de strings (rutas de los juegos)
        # split puede ser "train", "valid", "test", "all" o una lista de ellos (dependiendo de la implementación interna)
        # Probemos con "all" primero o iterando.
        
        all_games = getListGames(split=["train", "valid", "test"]) 
        
        print(f"\nTotal de juegos encontrados: {len(all_games)}")
        print("-" * 50)
        print("Primeros 10 juegos:")
        print("-" * 50)
        
        # Imprimir los primeros 10 para ver el formato
        for game in all_games[:10]: 
            print(game)
            
    except Exception as e:
        print(f"Error al obtener los juegos: {e}")

if __name__ == "__main__":
    main()
