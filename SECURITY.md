# Security -- Averyn Face Service

## Datos biometricos

Los embeddings faciales (vectores de 512 floats generados por ArcFace) son
**datos personales sensibles** segun la mayoria de marcos legales (GDPR Art. 9,
leyes de proteccion de datos personales en LATAM).

### Donde se almacenan

- **Face service**: NO almacena nada. Es stateless. Los embeddings se calculan,
  se devuelven al caller (NestJS gateway) y se descartan de memoria.
- **Gateway (NestJS)**: recibe el embedding y lo almacena en PostgreSQL (Supabase)
  en la tabla de perfiles, columna `face_embedding vector(512)`.

### Quien puede leer

- **Supabase RLS** (Row Level Security): solo el service role del backend puede
  acceder a la columna `face_embedding`. Los clientes no tienen acceso directo.
- **Face service**: solo recibe embeddings via HTTP interno (red Docker privada),
  nunca los persiste ni los loguea.

### Como se borra

- **Endpoint de borrado en NestJS**: el usuario solicita eliminacion de su cuenta.
  NestJS ejecuta `DELETE FROM profiles WHERE id = $1` (Supabase). El embedding
  se borra junto con el resto del perfil.
- **Derecho al borrado**: el sistema debe soportar la eliminacion completa del
  embedding en respuesta a una solicitud del titular de los datos.

## Proteccion en transit

- **Red interna Docker**: el face-service solo escucha en la red interna de
  Docker Compose. **Nunca** se expone al puerto publico del VPS.
- **TLS externo**: el reverse proxy (Caddy/nginx) maneja TLS para el trafico
  publico hacia NestJS.
- **TLS interno (opcional)**: si la politica de datos sensibles lo exige,
  configurar mTLS entre NestJS y face-service dentro de Docker.

## Proteccion en reposo

- **Embeddings en Supabase**: cifrado en reposo gestionado por Supabase
  (AWS RDS encryption at rest). Opcionalmente, cifrado a nivel de columna
  con `pgcrypto` para defense-in-depth.
- **Face service**: no tiene persistencia, no hay datos en reposo que cifrar.

## Auditoria y logging

- **No loguear embeddings completos**: el servicio actual no loguea embeddings
  ni imagenes. Los logs JSON solo contienen metadatos (tiempos, resultados).
- **No loguear imagenes**: nunca se escriben archivos de imagen al disco.
  Las imagenes se procesan en memoria y se descartan.
- **Request ID**: cada request se asocia a un `request_id` para correlacionar
  en logs distribuidos (pendiente: integrar con NestJS gateway).

## Exposicion del servicio

- Puerto 8001: **solo red interna Docker**. No exponer al host ni a internet.
- Si se necesita acceso externo para debugging: tunel SSH o VPN, nunca
  el puerto directo.
- No hay autenticacion actualmente. Pendiente: API key interna
  gateway <-> face-service.

## Checklist pre-produccion

- [ ] Confirmar que el face-service nunca loguea imagenes ni embeddings
- [ ] Confirmar que el puerto 8001 no esta expuesto al exterior
- [ ] Implementar API key interna (gateway -> face-service)
- [ ] Configurar RLS en Supabase para la columna face_embedding
- [ ] Documentar flujo de borrado de embeddings en NestJS
- [ ] Auditar que no hay archivos sensibles en el repositorio git
- [ ] Confirmar que los logs no contienen datos biometricos
