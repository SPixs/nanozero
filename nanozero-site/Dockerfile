# Build en deux étapes : compile (JDK 25 + Maven) puis runtime (JRE 25 slim).
FROM maven:3.9-eclipse-temurin-25 AS build
WORKDIR /build
COPY pom.xml .
RUN mvn -B -q dependency:go-offline
COPY src ./src
RUN mvn -B -q clean package -DskipTests

FROM eclipse-temurin:25-jre-alpine
WORKDIR /app
RUN apk add --no-cache curl
COPY --from=build /build/target/nanozero-site.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
