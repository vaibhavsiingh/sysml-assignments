    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            A[i*M+j] = dist(gen);            
        }
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            B[i*K+j] = dist(gen);            
        }
    }

    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            A[i*M+j] = i+j;            
        }
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            B[i*K+j] = i+j;            
        }
    }