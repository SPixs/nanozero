package org.nanozero.site.game;

import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface GameRepository extends JpaRepository<Game, Long> {
  Optional<Game> findByShareId(String shareId);

  boolean existsByShareId(String shareId);
}
