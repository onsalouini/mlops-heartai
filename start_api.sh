#!/bin/bash
# Arrêter et supprimer l'ancien conteneur s'il existe
docker stop heartai 2>/dev/null
docker rm heartai 2>/dev/null

# Lancer le nouveau conteneur
docker run -d -p 8000:8000 --name heartai prenom_nom_classe_mlops

# Attendre que le serveur démarre
sleep 2

# Tester
echo "Test health endpoint:"
curl http://127.0.0.1:8000/health

echo -e "\n\nTest prediction endpoint:"
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]}'
