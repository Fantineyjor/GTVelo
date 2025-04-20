from .trainer import train_vae
 
def train(model, adata, epochs = 50, learning_rate = 1e-2, batch_size = 200, grad_clip = 1000, shuffle=True, test=0.1, name = '', optimizer='adam', random_seed=42):

    epochs, val_ae, val_traj = train_vae(model, adata, epochs=epochs,
                learning_rate=learning_rate, batch_size=batch_size,
                grad_clip=grad_clip, shuffle=shuffle, test=test, name=name,
                optimizer=optimizer, random_seed=random_seed)
    return epochs, val_ae, val_traj
