--
-- HuggingArch Supabase fresh-install baseline
-- ============================================
--
-- This is the complete v1 database shape: all public tables, views,
-- functions, indexes, constraints, RLS policies, and grants through migration
-- 0041, with later ADD COLUMN migrations folded into their CREATE TABLE
-- definitions; and, from the numbering that restarted at 0000, every
-- migration through 0020: the Kernel Bench objects through 0026 (the Postgres
-- measurement store is gone, the measurements live in a DuckLake,
-- docs/measured_calibration.md §4, and only the lake lease of 0023 --
-- bench_lake_lease, bench_lake_claim, bench_lake_transfer -- remains), and
-- the spec-generation migrations 0000, 0002, 0003, 0005, 0008 and 0020
-- (folded 2026-10-09). Run this ONE file on a NEW, EMPTY Supabase project. Do
-- not run the historical migrations afterwards: they are upgrades for
-- existing installations and would redo work that this baseline has already
-- folded in.
--
-- The body is `pg_dump --schema-only --no-owner -n public` of the live,
-- fully-migrated database. It assumes what a Supabase project provides: the
-- roles anon, authenticated, service_role and supabase_admin (grants and
-- default privileges name them) and the auth schema (auth.users, auth.uid()),
-- which functions and RLS policies reference.
--
-- Recommended:
--   psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f supabase/init.sql
--
-- The script is deliberately not an upgrade script. It refuses to initialize a
-- database that already has public.profiles. Existing HuggingArch databases
-- should continue from the numbered migration after the baseline they last
-- applied.
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

DO $do$
BEGIN
  IF to_regclass('public.profiles') IS NOT NULL THEN
    RAISE EXCEPTION 'HuggingArch schema already exists; init.sql is for a new, empty Supabase project';
  END IF;
END
$do$;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA IF NOT EXISTS public;


--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA public IS 'standard public schema';


--
-- Name: add_admin(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.add_admin(p_user_id uuid) RETURNS boolean
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  current_admin_id UUID;
BEGIN
  current_admin_id := auth.uid();
  IF NOT EXISTS (SELECT 1 FROM public.admins WHERE user_id = current_admin_id) THEN
    RAISE EXCEPTION 'Only admins can add other admins';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM auth.users WHERE id = p_user_id) THEN
    RAISE EXCEPTION 'User does not exist';
  END IF;
  INSERT INTO public.admins (user_id, created_by)
  VALUES (p_user_id, current_admin_id)
  ON CONFLICT (user_id) DO NOTHING;
  RETURN TRUE;
END;
$$;


--
-- Name: bench_lake_claim(uuid, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bench_lake_claim(p_lake uuid, p_machine text) RETURNS text
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public'
    AS $$
DECLARE
    held_lake uuid;
    held_machine text;
BEGIN
    IF p_lake IS NULL OR coalesce(p_machine, '') = '' THEN
        RAISE EXCEPTION 'bench_lake_claim: p_lake and p_machine are required';
    END IF;
    INSERT INTO public.bench_lake_lease (id, lake_id, machine, claimed_at)
    VALUES (true, p_lake, p_machine, now())
    ON CONFLICT (id) DO NOTHING;
    IF FOUND THEN
        RETURN 'claimed';
    END IF;
    -- The row exists (a concurrent first claim waited on the key above and
    -- lands here): lock it, then compare both.
    SELECT lake_id, machine INTO held_lake, held_machine
      FROM public.bench_lake_lease WHERE id FOR UPDATE;
    IF held_lake = p_lake AND held_machine = p_machine THEN
        UPDATE public.bench_lake_lease SET claimed_at = now() WHERE id;
        RETURN 'ok';
    END IF;
    RETURN 'refused';
END
$$;


--
-- Name: bench_lake_transfer(text, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bench_lake_transfer(p_old text, p_new text) RETURNS text
    LANGUAGE plpgsql
    SET search_path TO 'public'
    AS $$
BEGIN
    IF coalesce(p_new, '') = '' THEN
        RAISE EXCEPTION 'bench_lake_transfer: p_new is required';
    END IF;
    UPDATE public.bench_lake_lease SET machine = p_new, claimed_at = now()
     WHERE id AND machine = p_old;
    IF FOUND THEN
        RETURN 'moved';
    END IF;
    RETURN 'refused';
END
$$;


--
-- Name: bump_cached_spec_hit(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bump_cached_spec_hit(p_model_id text) RETURNS void
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  UPDATE public.cached_specs
  SET hit_count    = hit_count + 1,
      last_used_at = NOW()
  WHERE model_id = p_model_id;
END;
$$;


--
-- Name: cleanup_stale_spec_locks(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.cleanup_stale_spec_locks() RETURNS integer
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  v_deleted INTEGER;
BEGIN
  WITH d AS (
    DELETE FROM public.spec_generation_locks
    WHERE expires_at < NOW()
    RETURNING 1
  )
  SELECT COUNT(*) INTO v_deleted FROM d;
  RETURN v_deleted;
END;
$$;


--
-- Name: create_user_profile_if_not_exists(uuid, text, text, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.create_user_profile_if_not_exists(p_user_id uuid, p_username text DEFAULT NULL::text, p_avatar_url text DEFAULT NULL::text, p_full_name text DEFAULT NULL::text) RETURNS void
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  v_username TEXT;
BEGIN
  IF EXISTS (SELECT 1 FROM public.profiles WHERE id = p_user_id) THEN
    RETURN;
  END IF;

  v_username := COALESCE(
    p_username,
    (SELECT split_part(email, '@', 1) FROM auth.users WHERE id = p_user_id LIMIT 1),
    'user'
  );

  INSERT INTO public.profiles (id, username, avatar_url, full_name)
  VALUES (p_user_id, v_username, p_avatar_url, p_full_name)
  ON CONFLICT (id) DO NOTHING;
EXCEPTION
  WHEN OTHERS THEN
    RAISE WARNING 'Failed to create profile for user %: %', p_user_id, SQLERRM;
END;
$$;


--
-- Name: delete_cached_spec(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.delete_cached_spec(p_model_id text) RETURNS boolean
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  v_deleted INTEGER;
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  WITH d AS (
    DELETE FROM public.cached_specs
    WHERE model_id = p_model_id
    RETURNING 1
  )
  SELECT COUNT(*) INTO v_deleted FROM d;
  RETURN v_deleted > 0;
END;
$$;


--
-- Name: delete_user_data(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.delete_user_data(p_user_id uuid) RETURNS boolean
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  current_admin_id UUID;
  target_is_admin BOOLEAN;
  admin_count INTEGER;
BEGIN
  current_admin_id := auth.uid();
  IF NOT EXISTS (SELECT 1 FROM public.admins WHERE user_id = current_admin_id) THEN
    RAISE EXCEPTION 'Only admins can delete users';
  END IF;
  IF p_user_id = current_admin_id THEN
    RAISE EXCEPTION 'Cannot delete yourself';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM auth.users WHERE id = p_user_id) THEN
    RAISE EXCEPTION 'User does not exist';
  END IF;

  SELECT EXISTS(SELECT 1 FROM public.admins WHERE user_id = p_user_id) INTO target_is_admin;
  IF target_is_admin THEN
    SELECT COUNT(*) INTO admin_count FROM public.admins;
    IF admin_count <= 1 THEN
      RAISE EXCEPTION 'Cannot delete the last admin';
    END IF;
  END IF;

  -- Cascading FK on profiles takes care of analyze_logs (user_id ON DELETE SET NULL),
  -- login_logs (CASCADE), user_models (CASCADE), admins (CASCADE).
  DELETE FROM public.admins  WHERE user_id = p_user_id;
  DELETE FROM public.profiles WHERE id      = p_user_id;
  RETURN TRUE;
END;
$$;


--
-- Name: get_active_users_stats(date, date); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_active_users_stats(p_start_date date DEFAULT (CURRENT_DATE - '30 days'::interval), p_end_date date DEFAULT CURRENT_DATE) RETURNS TABLE(date date, new_users integer, active_users integer, login_count integer, avg_login_per_user numeric)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  WITH daily_logs AS (
    SELECT
      DATE(ll.login_at) as log_date,
      ll.user_id,
      NOT EXISTS (
        SELECT 1 FROM public.profiles p
        WHERE p.id = ll.user_id AND p.created_at <= ll.login_at
      ) as is_new_user
    FROM public.login_logs ll
    WHERE DATE(ll.login_at) BETWEEN p_start_date AND p_end_date
  )
  SELECT
    log_date as date,
    COUNT(DISTINCT user_id) FILTER (WHERE is_new_user)::INTEGER as new_users,
    COUNT(DISTINCT user_id)::INTEGER as active_users,
    COUNT(*)::INTEGER as login_count,
    CASE WHEN COUNT(DISTINCT user_id) > 0
         THEN ROUND(COUNT(*)::NUMERIC / COUNT(DISTINCT user_id), 2)
         ELSE 0 END as avg_login_per_user
  FROM daily_logs
  GROUP BY log_date
  ORDER BY log_date DESC;
END;
$$;


--
-- Name: get_active_users_today(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_active_users_today() RETURNS integer
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN (
    SELECT COUNT(DISTINCT user_id)::INTEGER
    FROM public.login_logs
    WHERE DATE(login_at) = CURRENT_DATE AND user_id IS NOT NULL
  );
END;
$$;


--
-- Name: get_cached_spec(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_cached_spec(p_model_id text) RETURNS TABLE(model_id text, spec_yaml text, snapshot_json jsonb, new_component_yamls jsonb, agent_cli text, hit_count integer, created_at timestamp with time zone, last_used_at timestamp with time zone)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  RETURN QUERY
  SELECT cs.model_id, cs.spec_yaml, cs.snapshot_json,
         cs.new_component_yamls, cs.agent_cli, cs.hit_count,
         cs.created_at, cs.last_used_at
  FROM public.cached_specs cs
  WHERE cs.model_id = p_model_id;
END;
$$;


--
-- Name: get_cached_spec_detail(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_cached_spec_detail(p_model_id text) RETURNS TABLE(model_id text, generated_by_email text, agent_cli text, hit_count integer, spec_yaml text, snapshot_json jsonb, new_component_yamls jsonb, created_at timestamp with time zone, last_used_at timestamp with time zone)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    cs.model_id,
    COALESCE(au.email::TEXT, '') AS generated_by_email,
    cs.agent_cli,
    cs.hit_count,
    cs.spec_yaml,
    cs.snapshot_json,
    cs.new_component_yamls,
    cs.created_at,
    cs.last_used_at
  FROM public.cached_specs cs
  LEFT JOIN auth.users au ON cs.generated_by = au.id
  WHERE cs.model_id = p_model_id;
END;
$$;


--
-- Name: get_cached_specs(integer, integer, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_cached_specs(p_limit integer DEFAULT 50, p_offset integer DEFAULT 0, p_sort text DEFAULT 'hits'::text) RETURNS TABLE(model_id text, generated_by_email text, agent_cli text, hit_count integer, spec_yaml_bytes integer, new_component_count integer, created_at timestamp with time zone, last_used_at timestamp with time zone)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    cs.model_id,
    COALESCE(au.email::TEXT, '') AS generated_by_email,
    cs.agent_cli,
    cs.hit_count,
    octet_length(cs.spec_yaml)                          AS spec_yaml_bytes,
    COALESCE(jsonb_array_length(cs.new_component_yamls), 0) AS new_component_count,
    cs.created_at,
    cs.last_used_at
  FROM public.cached_specs cs
  LEFT JOIN auth.users au ON cs.generated_by = au.id
  ORDER BY
    CASE WHEN p_sort = 'hits'   THEN cs.hit_count    END DESC NULLS LAST,
    CASE WHEN p_sort = 'recent' THEN cs.last_used_at END DESC NULLS LAST,
    CASE WHEN p_sort = 'oldest' THEN cs.created_at   END ASC  NULLS LAST,
    cs.model_id  -- stable tie-breaker
  LIMIT p_limit
  OFFSET p_offset;
END;
$$;


--
-- Name: get_cached_specs_count(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_cached_specs_count() RETURNS integer
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN (SELECT COUNT(*)::INTEGER FROM public.cached_specs);
END;
$$;


--
-- Name: get_daily_call_trend(date, date); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_daily_call_trend(p_start_date date DEFAULT (CURRENT_DATE - '30 days'::interval), p_end_date date DEFAULT CURRENT_DATE) RETURNS TABLE(day date, analyze_calls integer, inference_calls integer, total_calls integer, unique_users integer, anon_calls integer)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    DATE(al.created_at) AS day,
    COUNT(*) FILTER (WHERE al.endpoint = 'analyze')::INTEGER             AS analyze_calls,
    COUNT(*) FILTER (WHERE al.endpoint = 'inference_estimate')::INTEGER  AS inference_calls,
    COUNT(*)::INTEGER                                                    AS total_calls,
    COUNT(DISTINCT al.user_id)::INTEGER                                  AS unique_users,
    COUNT(*) FILTER (WHERE al.user_id IS NULL)::INTEGER                  AS anon_calls
  FROM public.analyze_logs al
  WHERE DATE(al.created_at) BETWEEN p_start_date AND p_end_date
  GROUP BY DATE(al.created_at)
  ORDER BY day;
END;
$$;


--
-- Name: get_inference_query_dist(date, date); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_inference_query_dist(p_start_date date DEFAULT (CURRENT_DATE - '30 days'::interval), p_end_date date DEFAULT CURRENT_DATE) RETURNS TABLE(gpu_sku text, query_count integer, avg_cluster_size numeric, unique_users integer)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    COALESCE(al.metadata->>'gpu_sku', 'unknown') as gpu_sku,
    COUNT(*)::INTEGER as query_count,
    ROUND(AVG((al.metadata->>'cluster_size')::numeric), 2) as avg_cluster_size,
    COUNT(DISTINCT al.user_id)::INTEGER as unique_users
  FROM public.analyze_logs al
  WHERE al.endpoint = 'inference_estimate'
    AND DATE(al.created_at) BETWEEN p_start_date AND p_end_date
  GROUP BY gpu_sku
  ORDER BY query_count DESC;
END;
$$;


--
-- Name: get_latency_summary(date, date); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_latency_summary(p_start_date date DEFAULT (CURRENT_DATE - '30 days'::interval), p_end_date date DEFAULT CURRENT_DATE) RETURNS TABLE(endpoint text, call_count integer, p50_ms numeric, p95_ms numeric, max_ms integer, avg_ms numeric)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    al.endpoint,
    COUNT(*)::INTEGER                                                            AS call_count,
    PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY al.duration_ms)                 AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY al.duration_ms)                 AS p95_ms,
    MAX(al.duration_ms)::INTEGER                                                 AS max_ms,
    ROUND(AVG(al.duration_ms)::NUMERIC, 1)                                       AS avg_ms
  FROM public.analyze_logs al
  WHERE DATE(al.created_at) BETWEEN p_start_date AND p_end_date
    AND al.duration_ms IS NOT NULL
  GROUP BY al.endpoint
  ORDER BY call_count DESC;
END;
$$;


--
-- Name: get_login_logs_with_email(integer, integer, date, date, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_login_logs_with_email(p_limit integer DEFAULT 100, p_offset integer DEFAULT 0, p_start_date date DEFAULT NULL::date, p_end_date date DEFAULT NULL::date, p_user_email text DEFAULT NULL::text) RETURNS TABLE(id uuid, user_id uuid, user_email text, login_at timestamp with time zone, ip_address inet, user_agent text)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    ll.id,
    ll.user_id,
    COALESCE(au.email::TEXT, 'N/A') as user_email,
    ll.login_at,
    ll.ip_address,
    ll.user_agent
  FROM public.login_logs ll
  LEFT JOIN auth.users au ON ll.user_id = au.id
  WHERE
    (p_start_date IS NULL OR DATE(ll.login_at) >= p_start_date)
    AND (p_end_date IS NULL OR DATE(ll.login_at) <= p_end_date)
    AND (p_user_email IS NULL OR p_user_email = '' OR au.email::TEXT ILIKE '%' || p_user_email || '%')
  ORDER BY ll.login_at DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$;


--
-- Name: get_spec_generation_detail(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_spec_generation_detail(p_id uuid) RETURNS TABLE(id uuid, model_id text, generated_by_email text, agent_cli text, status text, spec_yaml text, snapshot_json jsonb, new_component_yamls jsonb, contributed_at timestamp with time zone, pr_url text, created_at timestamp with time zone, engine_gaps jsonb, repairable_errors jsonb, blocked_engine_version text, run_started_at timestamp with time zone, revision bigint)
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    sg.id,
    sg.model_id,
    COALESCE(au.email::TEXT, '') AS generated_by_email,
    sg.agent_cli,
    sg.status,
    sg.spec_yaml,
    sg.snapshot_json,
    sg.new_component_yamls,
    sg.contributed_at,
    sg.pr_url,
    sg.created_at,
    CASE WHEN sg.status = 'blocked_by_engine' THEN sg.pending_detail END,
    CASE WHEN sg.status = 'needs_repair' THEN sg.pending_detail END,
    sg.blocked_engine_version,
    sg.run_started_at,
    sg.revision
  FROM public.spec_generations sg
  LEFT JOIN auth.users au ON sg.user_id = au.id
  WHERE sg.id = p_id;
END;
$$;


--
-- Name: get_spec_generation_reviews(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_spec_generation_reviews(p_record_id uuid) RETURNS TABLE(id uuid, record_id uuid, model_id text, reviewer_email text, reviewer_role text, verdict text, report text, agent_cli text, created_at timestamp with time zone, bundle_fingerprint text, run_revision text, engine_version text, admission jsonb)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    sr.id,
    sr.record_id,
    sr.model_id,
    COALESCE(au.email::TEXT, '') AS reviewer_email,
    sr.reviewer_role,
    sr.verdict,
    sr.report,
    sr.agent_cli,
    sr.created_at,
    sr.bundle_fingerprint,
    sr.run_revision,
    sr.engine_version,
    sr.admission
  FROM public.spec_generation_reviews sr
  LEFT JOIN auth.users au ON sr.reviewer_id = au.id
  WHERE sr.record_id = p_record_id
  ORDER BY sr.created_at DESC, sr.id DESC;
END;
$$;


--
-- Name: get_spec_generation_runs(integer, integer, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_spec_generation_runs(p_limit integer DEFAULT 50, p_offset integer DEFAULT 0, p_error_kind text DEFAULT NULL::text) RETURNS TABLE(id uuid, model_id text, user_email text, kind text, state text, success boolean, error_kind text, engine_gaps jsonb, message text, created_at timestamp with time zone, finished_at timestamp with time zone)
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  IF p_error_kind IS NOT NULL AND NOT (p_error_kind = ANY (ARRAY[
      'spec_invalid', 'spec_bundle_incomplete', 'spec_needs_declaration',
      'bundle_invalid', 'blocked_by_engine', 'analysis_refusal_unattributed',
      'agent_unavailable', 'timed_out', 'killed', 'context_exhausted'])) THEN
    RAISE EXCEPTION 'unknown run error kind %', p_error_kind USING ERRCODE = '22023';
  END IF;
  RETURN QUERY
  SELECT
    l.id,
    l.model_id,
    COALESCE(au.email::TEXT, '') AS user_email,
    l.kind,
    l.state,
    l.success,
    l.error_kind,
    l.engine_gaps,
    l.message,
    l.created_at,
    l.finished_at
  FROM public.spec_generation_logs l
  LEFT JOIN auth.users au ON l.user_id = au.id
  WHERE p_error_kind IS NULL OR l.error_kind = p_error_kind
  ORDER BY l.created_at DESC, l.id DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$;


--
-- Name: get_spec_generation_runs_count(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_spec_generation_runs_count(p_error_kind text DEFAULT NULL::text) RETURNS integer
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  IF p_error_kind IS NOT NULL AND NOT (p_error_kind = ANY (ARRAY[
      'spec_invalid', 'spec_bundle_incomplete', 'spec_needs_declaration',
      'bundle_invalid', 'blocked_by_engine', 'analysis_refusal_unattributed',
      'agent_unavailable', 'timed_out', 'killed', 'context_exhausted'])) THEN
    RAISE EXCEPTION 'unknown run error kind %', p_error_kind USING ERRCODE = '22023';
  END IF;
  RETURN (SELECT COUNT(*)::INTEGER FROM public.spec_generation_logs l
          WHERE p_error_kind IS NULL OR l.error_kind = p_error_kind);
END;
$$;


--
-- Name: get_spec_generations(integer, integer, text, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_spec_generations(p_limit integer DEFAULT 50, p_offset integer DEFAULT 0, p_sort text DEFAULT 'recent'::text, p_status text DEFAULT NULL::text) RETURNS TABLE(id uuid, model_id text, generated_by_email text, agent_cli text, status text, spec_yaml_bytes integer, new_component_count integer, contributed_at timestamp with time zone, created_at timestamp with time zone, engine_gaps jsonb, blocked_engine_version text)
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  IF p_status IS NOT NULL AND NOT (p_status = ANY (ARRAY[
      'generated', 'contributed', 'merged', 'rejected', 'closed',
      'blocked_by_engine', 'needs_repair'])) THEN
    RAISE EXCEPTION 'unknown generation status %', p_status USING ERRCODE = '22023';
  END IF;
  RETURN QUERY
  SELECT
    sg.id,
    sg.model_id,
    COALESCE(au.email::TEXT, '') AS generated_by_email,
    sg.agent_cli,
    sg.status,
    octet_length(sg.spec_yaml)                             AS spec_yaml_bytes,
    COALESCE(jsonb_array_length(sg.new_component_yamls), 0) AS new_component_count,
    sg.contributed_at,
    sg.created_at,
    CASE WHEN sg.status = 'blocked_by_engine' THEN sg.pending_detail END,
    sg.blocked_engine_version
  FROM public.spec_generations sg
  LEFT JOIN auth.users au ON sg.user_id = au.id
  WHERE p_status IS NULL OR sg.status = p_status
  ORDER BY
    CASE WHEN p_sort = 'recent' THEN sg.created_at END DESC NULLS LAST,
    CASE WHEN p_sort = 'oldest' THEN sg.created_at END ASC  NULLS LAST,
    CASE WHEN p_sort = 'model'  THEN sg.model_id   END ASC  NULLS LAST,
    sg.created_at DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$;


--
-- Name: get_spec_generations_count(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_spec_generations_count(p_status text DEFAULT NULL::text) RETURNS integer
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  IF p_status IS NOT NULL AND NOT (p_status = ANY (ARRAY[
      'generated', 'contributed', 'merged', 'rejected', 'closed',
      'blocked_by_engine', 'needs_repair'])) THEN
    RAISE EXCEPTION 'unknown generation status %', p_status USING ERRCODE = '22023';
  END IF;
  RETURN (SELECT COUNT(*)::INTEGER FROM public.spec_generations sg
          WHERE p_status IS NULL OR sg.status = p_status);
END;
$$;


--
-- Name: get_top_models(date, date, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_top_models(p_start_date date DEFAULT (CURRENT_DATE - '30 days'::interval), p_end_date date DEFAULT CURRENT_DATE, p_limit integer DEFAULT 20) RETURNS TABLE(model_id text, total_calls integer, analyze_calls integer, inference_calls integer, unique_users integer)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    al.model_id,
    COUNT(*)::INTEGER as total_calls,
    COUNT(*) FILTER (WHERE al.endpoint = 'analyze')::INTEGER as analyze_calls,
    COUNT(*) FILTER (WHERE al.endpoint = 'inference_estimate')::INTEGER as inference_calls,
    COUNT(DISTINCT al.user_id)::INTEGER as unique_users
  FROM public.analyze_logs al
  WHERE DATE(al.created_at) BETWEEN p_start_date AND p_end_date
  GROUP BY al.model_id
  ORDER BY total_calls DESC
  LIMIT p_limit;
END;
$$;


--
-- Name: get_total_analyze_logs_count(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_total_analyze_logs_count() RETURNS integer
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN (SELECT COUNT(*)::INTEGER FROM public.analyze_logs);
END;
$$;


--
-- Name: get_total_login_logs_count(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_total_login_logs_count() RETURNS integer
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN (SELECT COUNT(*)::INTEGER FROM public.login_logs);
END;
$$;


--
-- Name: get_user_analyze_logs(uuid, integer, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_user_analyze_logs(p_user_id uuid, p_limit integer DEFAULT 50, p_offset integer DEFAULT 0) RETURNS TABLE(id uuid, endpoint text, model_id text, duration_ms integer, metadata jsonb, created_at timestamp with time zone)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT al.id, al.endpoint, al.model_id, al.duration_ms,
         al.metadata, al.created_at
  FROM public.analyze_logs al
  WHERE al.user_id = p_user_id
  ORDER BY al.created_at DESC
  LIMIT p_limit OFFSET p_offset;
END;
$$;


--
-- Name: get_user_summary(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_user_summary(p_user_id uuid) RETURNS TABLE(total_calls integer, analyze_calls integer, inference_calls integer, unique_models integer, first_event_at timestamp with time zone, last_event_at timestamp with time zone, login_count integer, last_login_at timestamp with time zone)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    COALESCE(COUNT(al.*)::INTEGER, 0)                                   AS total_calls,
    COUNT(*) FILTER (WHERE al.endpoint = 'analyze')::INTEGER            AS analyze_calls,
    COUNT(*) FILTER (WHERE al.endpoint = 'inference_estimate')::INTEGER AS inference_calls,
    COUNT(DISTINCT al.model_id)::INTEGER                                AS unique_models,
    MIN(al.created_at)                                                  AS first_event_at,
    MAX(al.created_at)                                                  AS last_event_at,
    (SELECT COUNT(*)::INTEGER FROM public.login_logs ll WHERE ll.user_id = p_user_id) AS login_count,
    (SELECT MAX(ll.login_at) FROM public.login_logs ll WHERE ll.user_id = p_user_id)  AS last_login_at
  FROM public.analyze_logs al
  WHERE al.user_id = p_user_id;
END;
$$;


--
-- Name: get_users_count(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_users_count() RETURNS integer
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN (SELECT COUNT(*)::INTEGER FROM public.profiles);
END;
$$;


--
-- Name: get_users_with_admin_status(integer, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_users_with_admin_status(p_limit integer DEFAULT 100, p_offset integer DEFAULT 0) RETURNS TABLE(id uuid, user_email text, username text, full_name text, avatar_url text, created_at timestamp with time zone, is_admin boolean, is_super boolean)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF NOT public.is_admin(auth.uid()) THEN
    RAISE EXCEPTION 'Admin access required';
  END IF;
  RETURN QUERY
  SELECT
    p.id,
    COALESCE(au.email::TEXT, 'N/A')                          AS user_email,
    p.username,
    p.full_name,
    p.avatar_url,
    p.created_at,
    EXISTS(SELECT 1 FROM public.admins a WHERE a.user_id = p.id)                 AS is_admin,
    EXISTS(SELECT 1 FROM public.admins a WHERE a.user_id = p.id AND a.is_super)  AS is_super
  FROM public.profiles p
  LEFT JOIN auth.users au ON p.id = au.id
  ORDER BY p.created_at DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$;


--
-- Name: is_admin(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.is_admin(check_user_id uuid DEFAULT auth.uid()) RETURNS boolean
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  IF check_user_id IS NULL THEN
    RETURN FALSE;
  END IF;
  RETURN EXISTS (
    SELECT 1 FROM public.admins WHERE user_id = check_user_id
  );
END;
$$;


--
-- Name: is_super_admin(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.is_super_admin(check_user_id uuid) RETURNS boolean
    LANGUAGE sql STABLE SECURITY DEFINER
    AS $$
  SELECT EXISTS(SELECT 1 FROM public.admins WHERE user_id = check_user_id AND is_super);
$$;


--
-- Name: log_user_login(uuid, inet, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.log_user_login(p_user_id uuid, p_ip_address inet DEFAULT NULL::inet, p_user_agent text DEFAULT NULL::text) RETURNS uuid
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  log_id UUID;
BEGIN
  INSERT INTO public.login_logs (user_id, ip_address, user_agent)
  VALUES (p_user_id, p_ip_address, p_user_agent)
  RETURNING id INTO log_id;
  RETURN log_id;
END;
$$;


--
-- Name: record_generation(uuid, text, text, jsonb, jsonb, text, text, timestamp with time zone); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.record_generation(p_user_id uuid, p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_run_started_at timestamp with time zone) RETURNS jsonb
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
DECLARE
  v_id uuid;
BEGIN
  IF p_user_id IS NULL OR p_model_id IS NULL OR p_run_started_at IS NULL THEN
    RAISE EXCEPTION 'record_generation needs a user, a model and a run start'
      USING ERRCODE = '22023';
  END IF;
  PERFORM pg_advisory_xact_lock(
    hashtextextended(p_user_id::text || '|' || p_model_id, 0));

  SELECT sg.id INTO v_id FROM public.spec_generations sg
   WHERE sg.user_id = p_user_id AND sg.model_id = p_model_id
     AND sg.status = ANY (ARRAY['generated', 'rejected', 'closed'])
   ORDER BY sg.created_at DESC
   LIMIT 1
   FOR UPDATE;

  IF v_id IS NOT NULL THEN
    UPDATE public.spec_generations SET
      spec_yaml = p_spec_yaml,
      snapshot_json = COALESCE(p_snapshot_json, '{}'::jsonb),
      new_component_yamls = COALESCE(p_new_component_yamls, '[]'::jsonb),
      agent_cli = p_agent_cli,
      agent_model = p_agent_model,
      status = 'generated',
      pr_url = NULL,
      created_at = now(),
      run_started_at = p_run_started_at,
      pending_detail = NULL,
      blocked_engine_version = NULL
    WHERE spec_generations.id = v_id;
  ELSE
    INSERT INTO public.spec_generations (
      user_id, model_id, spec_yaml, snapshot_json, new_component_yamls,
      agent_cli, agent_model, status, run_started_at)
    VALUES (
      p_user_id, p_model_id, p_spec_yaml,
      COALESCE(p_snapshot_json, '{}'::jsonb),
      COALESCE(p_new_component_yamls, '[]'::jsonb),
      p_agent_cli, p_agent_model, 'generated', p_run_started_at)
    RETURNING spec_generations.id INTO v_id;
  END IF;

  DELETE FROM public.spec_generations sg
   WHERE sg.user_id = p_user_id AND sg.model_id = p_model_id
     AND sg.status = ANY (ARRAY['blocked_by_engine', 'needs_repair'])
     AND COALESCE(sg.run_started_at, sg.created_at) < p_run_started_at;

  RETURN jsonb_build_object('outcome', 'recorded', 'id', v_id);
END;
$$;


--
-- Name: record_pending_generation(uuid, text, text, jsonb, jsonb, text, text, jsonb, text, timestamp with time zone); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.record_pending_generation(p_user_id uuid, p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_engine_gaps jsonb, p_engine_version text, p_run_started_at timestamp with time zone) RETURNS jsonb
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
DECLARE
  v_id uuid;
BEGIN
  IF p_user_id IS NULL OR p_model_id IS NULL OR p_run_started_at IS NULL THEN
    RAISE EXCEPTION 'record_pending_generation needs a user, a model and a run start'
      USING ERRCODE = '22023';
  END IF;
  PERFORM pg_advisory_xact_lock(
    hashtextextended(p_user_id::text || '|' || p_model_id, 0));

  IF EXISTS (
    SELECT 1 FROM public.spec_generations sg
     WHERE sg.user_id = p_user_id AND sg.model_id = p_model_id
       AND COALESCE(sg.run_started_at, sg.created_at) >= p_run_started_at
  ) THEN
    RETURN jsonb_build_object('outcome', 'stale');
  END IF;

  INSERT INTO public.spec_generations AS sg (
    user_id, model_id, spec_yaml, snapshot_json, new_component_yamls,
    agent_cli, agent_model, status, pending_detail, blocked_engine_version,
    run_started_at)
  VALUES (
    p_user_id, p_model_id, p_spec_yaml,
    COALESCE(p_snapshot_json, '{}'::jsonb),
    COALESCE(p_new_component_yamls, '[]'::jsonb),
    p_agent_cli, p_agent_model, 'blocked_by_engine', p_engine_gaps,
    p_engine_version, p_run_started_at)
  ON CONFLICT (user_id, model_id)
    WHERE (status = ANY (ARRAY['blocked_by_engine', 'needs_repair']))
  DO UPDATE SET
    spec_yaml = EXCLUDED.spec_yaml,
    snapshot_json = EXCLUDED.snapshot_json,
    new_component_yamls = EXCLUDED.new_component_yamls,
    agent_cli = EXCLUDED.agent_cli,
    agent_model = EXCLUDED.agent_model,
    status = 'blocked_by_engine',
    pending_detail = EXCLUDED.pending_detail,
    blocked_engine_version = EXCLUDED.blocked_engine_version,
    run_started_at = EXCLUDED.run_started_at,
    created_at = now()
  WHERE sg.status = ANY (ARRAY['blocked_by_engine', 'needs_repair'])
    AND COALESCE(sg.run_started_at, sg.created_at) < EXCLUDED.run_started_at
  RETURNING sg.id INTO v_id;

  IF v_id IS NULL THEN
    RETURN jsonb_build_object('outcome', 'stale');
  END IF;
  RETURN jsonb_build_object('outcome', 'recorded', 'id', v_id);
END;
$$;


--
-- Name: release_spec_lock(text, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.release_spec_lock(p_model_id text, p_host_id text) RETURNS boolean
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  v_deleted INTEGER;
BEGIN
  WITH d AS (
    DELETE FROM public.spec_generation_locks
    WHERE model_id = p_model_id
      AND host_id  = p_host_id
      AND boot_id  = ''
    RETURNING 1
  )
  SELECT COUNT(*) INTO v_deleted FROM d;
  RETURN v_deleted > 0;
END;
$$;


--
-- Name: release_spec_lock_v2(text, text, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.release_spec_lock_v2(p_model_id text, p_host_id text, p_boot_id text) RETURNS boolean
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  v_deleted INTEGER;
BEGIN
  WITH d AS (
    DELETE FROM public.spec_generation_locks
    WHERE model_id = p_model_id
      AND host_id  = p_host_id
      AND boot_id  = p_boot_id
    RETURNING 1
  )
  SELECT COUNT(*) INTO v_deleted FROM d;
  RETURN v_deleted > 0;
END;
$$;


--
-- Name: remove_admin(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.remove_admin(p_user_id uuid) RETURNS boolean
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  current_admin_id UUID;
  admin_count INTEGER;
BEGIN
  current_admin_id := auth.uid();
  IF NOT EXISTS (SELECT 1 FROM public.admins WHERE user_id = current_admin_id) THEN
    RAISE EXCEPTION 'Only admins can remove other admins';
  END IF;
  SELECT COUNT(*) INTO admin_count FROM public.admins;
  IF admin_count <= 1 THEN
    RAISE EXCEPTION 'Cannot remove the last admin';
  END IF;
  IF p_user_id = current_admin_id THEN
    RAISE EXCEPTION 'Cannot remove yourself as admin';
  END IF;
  IF public.is_super_admin(p_user_id) AND NOT public.is_super_admin(current_admin_id) THEN
    RAISE EXCEPTION 'Only a super admin can remove a super admin';
  END IF;
  DELETE FROM public.admins WHERE user_id = p_user_id;
  RETURN TRUE;
END;
$$;


--
-- Name: set_super_admin(uuid, boolean); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.set_super_admin(p_user_id uuid, p_is_super boolean) RETURNS boolean
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  current_admin_id UUID;
  super_count INTEGER;
BEGIN
  current_admin_id := auth.uid();
  IF NOT public.is_super_admin(current_admin_id) THEN
    RAISE EXCEPTION 'Only a super admin can change super status';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM public.admins WHERE user_id = p_user_id) THEN
    RAISE EXCEPTION 'Target is not an admin';
  END IF;
  IF NOT p_is_super THEN
    IF p_user_id = current_admin_id THEN
      RAISE EXCEPTION 'Cannot revoke your own super status';
    END IF;
    SELECT COUNT(*) INTO super_count FROM public.admins WHERE is_super;
    IF super_count <= 1 THEN
      RAISE EXCEPTION 'Cannot remove the last super admin';
    END IF;
  END IF;
  UPDATE public.admins SET is_super = p_is_super WHERE user_id = p_user_id;
  RETURN TRUE;
END;
$$;


--
-- Name: spec_generations_bump_revision(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.spec_generations_bump_revision() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path TO 'public', 'pg_temp'
    AS $$
BEGIN
  IF OLD.status = ANY (ARRAY['blocked_by_engine', 'needs_repair'])
     AND NOT (NEW.status = ANY (ARRAY['blocked_by_engine', 'needs_repair']))
     AND COALESCE(current_setting('huggingarch.promoting_pending', true), '') <> OLD.id::text THEN
    RAISE EXCEPTION 'a pending generation record leaves pending only by promotion'
      USING ERRCODE = '23514';
  END IF;
  NEW.revision := OLD.revision + 1;
  RETURN NEW;
END;
$$;


--
-- Name: sweep_cache_clear(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.sweep_cache_clear() RETURNS void
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  DELETE FROM public.sweep_cache;
END;
$$;


--
-- Name: sweep_cache_get(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.sweep_cache_get(p_key text) RETURNS TABLE(payload jsonb)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  UPDATE public.sweep_cache
    SET hit_count = hit_count + 1, last_used_at = NOW()
    WHERE key = p_key;
  RETURN QUERY
    SELECT sc.payload FROM public.sweep_cache sc WHERE sc.key = p_key;
END;
$$;


--
-- Name: sweep_cache_upsert(text, jsonb); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.sweep_cache_upsert(p_key text, p_payload jsonb) RETURNS void
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  c_cap CONSTANT INTEGER := 5000;
BEGIN
  INSERT INTO public.sweep_cache (key, payload, hit_count, created_at, last_used_at)
  VALUES (p_key, p_payload, 0, NOW(), NOW())
  ON CONFLICT (key) DO UPDATE
    SET payload = EXCLUDED.payload, last_used_at = NOW();

  -- Prune only when over cap; delete the oldest-by-last_used surplus.
  IF (SELECT COUNT(*) FROM public.sweep_cache) > c_cap THEN
    DELETE FROM public.sweep_cache
    WHERE key IN (
      SELECT key FROM public.sweep_cache
      ORDER BY last_used_at ASC
      LIMIT (SELECT COUNT(*) - c_cap FROM public.sweep_cache)
    );
  END IF;
END;
$$;


--
-- Name: transition_pending_generation(uuid, bigint, text, jsonb, jsonb, text, text, jsonb, text, text, timestamp with time zone); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.transition_pending_generation(p_id uuid, p_expected_revision bigint, p_to text, p_engine_gaps jsonb DEFAULT NULL::jsonb, p_repairable_errors jsonb DEFAULT NULL::jsonb, p_engine_version text DEFAULT NULL::text, p_spec_yaml text DEFAULT NULL::text, p_new_component_yamls jsonb DEFAULT NULL::jsonb, p_agent_cli text DEFAULT NULL::text, p_agent_model text DEFAULT NULL::text, p_run_started_at timestamp with time zone DEFAULT NULL::timestamp with time zone) RETURNS jsonb
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
DECLARE
  v_user uuid;
  v_model text;
  v_row public.spec_generations%ROWTYPE;
  v_at timestamp with time zone;
  v_mutable uuid;
BEGIN
  IF p_to IS NULL OR NOT (p_to = ANY (ARRAY[
      'promote', 'blocked_by_engine', 'needs_repair'])) THEN
    RAISE EXCEPTION 'unknown pending transition %', p_to USING ERRCODE = '22023';
  END IF;

  SELECT sg.user_id, sg.model_id INTO v_user, v_model
    FROM public.spec_generations sg WHERE sg.id = p_id;
  IF NOT FOUND THEN
    RETURN jsonb_build_object('outcome', 'conflict');
  END IF;
  PERFORM pg_advisory_xact_lock(
    hashtextextended(v_user::text || '|' || v_model, 0));

  SELECT * INTO v_row FROM public.spec_generations sg
   WHERE sg.id = p_id FOR UPDATE;
  IF NOT FOUND
     OR v_row.revision <> p_expected_revision
     OR NOT (v_row.status = ANY (ARRAY['blocked_by_engine', 'needs_repair'])) THEN
    RETURN jsonb_build_object('outcome', 'conflict');
  END IF;

  IF p_to <> 'promote' THEN
    UPDATE public.spec_generations SET
      status = p_to,
      pending_detail = CASE
        WHEN p_to = 'blocked_by_engine' THEN p_engine_gaps
        WHEN p_to = 'needs_repair' THEN p_repairable_errors
      END,
      blocked_engine_version = CASE WHEN p_to = 'blocked_by_engine' THEN p_engine_version END
    WHERE spec_generations.id = p_id;
    RETURN jsonb_build_object('outcome', 'recorded', 'id', p_id, 'status', p_to);
  END IF;

  v_at := COALESCE(p_run_started_at, v_row.run_started_at, v_row.created_at);
  IF EXISTS (
    SELECT 1 FROM public.spec_generations sg
     WHERE sg.user_id = v_user AND sg.model_id = v_model
       AND sg.status = ANY (ARRAY['generated', 'contributed', 'merged', 'closed'])
       AND COALESCE(sg.run_started_at, sg.created_at) > v_at
  ) THEN
    DELETE FROM public.spec_generations WHERE spec_generations.id = p_id;
    RETURN jsonb_build_object('outcome', 'superseded');
  END IF;

  SELECT sg.id INTO v_mutable FROM public.spec_generations sg
   WHERE sg.user_id = v_user AND sg.model_id = v_model
     AND sg.status = ANY (ARRAY['generated', 'rejected', 'closed'])
   ORDER BY sg.created_at DESC
   LIMIT 1
   FOR UPDATE;

  IF v_mutable IS NULL THEN
    PERFORM set_config('huggingarch.promoting_pending', p_id::text, true);
    UPDATE public.spec_generations SET
      spec_yaml = COALESCE(p_spec_yaml, v_row.spec_yaml),
      new_component_yamls = COALESCE(p_new_component_yamls, v_row.new_component_yamls),
      agent_cli = COALESCE(p_agent_cli, v_row.agent_cli),
      agent_model = COALESCE(p_agent_model, v_row.agent_model),
      status = 'generated',
      pr_url = NULL,
      created_at = now(),
      run_started_at = v_at,
      pending_detail = NULL,
      blocked_engine_version = NULL
    WHERE spec_generations.id = p_id;
    PERFORM set_config('huggingarch.promoting_pending', '', true);
    RETURN jsonb_build_object('outcome', 'recorded', 'id', p_id);
  END IF;

  UPDATE public.spec_generations SET
    spec_yaml = COALESCE(p_spec_yaml, v_row.spec_yaml),
    snapshot_json = v_row.snapshot_json,
    new_component_yamls = COALESCE(p_new_component_yamls, v_row.new_component_yamls),
    agent_cli = COALESCE(p_agent_cli, v_row.agent_cli),
    agent_model = COALESCE(p_agent_model, v_row.agent_model),
    status = 'generated',
    pr_url = NULL,
    created_at = now(),
    run_started_at = v_at,
    pending_detail = NULL,
    blocked_engine_version = NULL
  WHERE spec_generations.id = v_mutable;
  DELETE FROM public.spec_generations WHERE spec_generations.id = p_id;
  RETURN jsonb_build_object('outcome', 'recorded', 'id', v_mutable);
END;
$$;


--
-- Name: try_acquire_spec_lock(text, text, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.try_acquire_spec_lock(p_model_id text, p_host_id text, p_ttl_seconds integer DEFAULT 1800) RETURNS TABLE(acquired boolean, held_by text, expires_at timestamp with time zone)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  v_now    TIMESTAMP WITH TIME ZONE := NOW();
  v_expiry TIMESTAMP WITH TIME ZONE := NOW() + (p_ttl_seconds || ' seconds')::INTERVAL;
BEGIN
  INSERT INTO public.spec_generation_locks (model_id, host_id, started_at, expires_at)
  VALUES (p_model_id, p_host_id, v_now, v_expiry)
  ON CONFLICT (model_id) DO UPDATE
    SET host_id    = EXCLUDED.host_id,
        started_at = EXCLUDED.started_at,
        expires_at = EXCLUDED.expires_at
    -- Only take over if the existing lock is stale.
    WHERE public.spec_generation_locks.expires_at < v_now;

  -- Now read what's actually in the row to tell the caller whether
  -- they own the lock or someone else does.
  RETURN QUERY
  SELECT (l.host_id = p_host_id),
         l.host_id,
         l.expires_at
  FROM public.spec_generation_locks l
  WHERE l.model_id = p_model_id;
END;
$$;


--
-- Name: try_acquire_spec_lock_v2(text, text, text, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.try_acquire_spec_lock_v2(p_model_id text, p_host_id text, p_boot_id text, p_ttl_seconds integer DEFAULT 1800) RETURNS TABLE(acquired boolean, held_by text, boot_id text, expires_at timestamp with time zone)
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
  v_now    TIMESTAMP WITH TIME ZONE := NOW();
  v_expiry TIMESTAMP WITH TIME ZONE := NOW() + (p_ttl_seconds || ' seconds')::INTERVAL;
BEGIN
  INSERT INTO public.spec_generation_locks (model_id, host_id, boot_id, started_at, expires_at)
  VALUES (p_model_id, p_host_id, p_boot_id, v_now, v_expiry)
  ON CONFLICT (model_id) DO UPDATE
    SET host_id    = EXCLUDED.host_id,
        boot_id    = EXCLUDED.boot_id,
        started_at = EXCLUDED.started_at,
        expires_at = EXCLUDED.expires_at
    WHERE public.spec_generation_locks.expires_at < v_now;

  RETURN QUERY
  SELECT (l.host_id = p_host_id AND l.boot_id = p_boot_id),
         l.host_id,
         l.boot_id,
         l.expires_at
  FROM public.spec_generation_locks l
  WHERE l.model_id = p_model_id;
END;
$$;


--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;


--
-- Name: upsert_cached_spec(text, text, jsonb, jsonb, text, uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.upsert_cached_spec(p_model_id text, p_spec_yaml text, p_snapshot_json jsonb DEFAULT '{}'::jsonb, p_new_component_yamls jsonb DEFAULT '[]'::jsonb, p_agent_cli text DEFAULT NULL::text, p_generated_by uuid DEFAULT NULL::uuid) RETURNS void
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
  INSERT INTO public.cached_specs (
    model_id, generated_by, spec_yaml, snapshot_json,
    new_component_yamls, agent_cli, hit_count, created_at, last_used_at
  )
  VALUES (
    p_model_id, p_generated_by, p_spec_yaml, p_snapshot_json,
    p_new_component_yamls, p_agent_cli, 1, NOW(), NOW()
  )
  ON CONFLICT (model_id) DO UPDATE
    SET spec_yaml           = EXCLUDED.spec_yaml,
        snapshot_json       = EXCLUDED.snapshot_json,
        new_component_yamls = EXCLUDED.new_component_yamls,
        agent_cli           = EXCLUDED.agent_cli,
        last_used_at        = NOW();
        -- intentionally do NOT reset hit_count or generated_by — those
        -- track usage history across regenerations.
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: admins; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.admins (
    user_id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    created_by uuid,
    is_super boolean DEFAULT false NOT NULL
);


--
-- Name: agent_task_defaults; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agent_task_defaults (
    agent_cli text NOT NULL,
    task text NOT NULL,
    model text NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: analyze_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.analyze_logs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid,
    endpoint text NOT NULL,
    model_id text NOT NULL,
    duration_ms integer,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: bench_lake_lease; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bench_lake_lease (
    id boolean DEFAULT true NOT NULL,
    lake_id uuid NOT NULL,
    machine text NOT NULL,
    claimed_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT bench_lake_lease_id_check CHECK (id)
);


--
-- Name: cached_specs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cached_specs (
    model_id text NOT NULL,
    generated_by uuid,
    spec_yaml text NOT NULL,
    snapshot_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    new_component_yamls jsonb DEFAULT '[]'::jsonb NOT NULL,
    agent_cli text,
    hit_count integer DEFAULT 1 NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    last_used_at timestamp with time zone DEFAULT now()
);


--
-- Name: login_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.login_logs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    ip_address inet,
    user_agent text,
    login_at timestamp with time zone DEFAULT now()
);


--
-- Name: personal_access_tokens; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.personal_access_tokens (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    name text DEFAULT ''::text NOT NULL,
    token_hash text NOT NULL,
    token_prefix text DEFAULT ''::text NOT NULL,
    scopes text DEFAULT 'bench'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    last_used_at timestamp with time zone,
    revoked_at timestamp with time zone
);


--
-- Name: profiles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.profiles (
    id uuid NOT NULL,
    username text,
    avatar_url text,
    full_name text,
    bio text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    github_handle text,
    agent_task_routing jsonb
);


--
-- Name: COLUMN profiles.github_handle; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.profiles.github_handle IS 'Optional GitHub username (no @). Used to build the Co-authored-by trailer when the user contributes a generated spec via the PR flow.';


--
-- Name: spec_generation_locks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.spec_generation_locks (
    model_id text NOT NULL,
    host_id text NOT NULL,
    started_at timestamp with time zone DEFAULT now(),
    expires_at timestamp with time zone NOT NULL,
    boot_id text DEFAULT ''::text NOT NULL
);


--
-- Name: COLUMN spec_generation_locks.host_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.spec_generation_locks.host_id IS 'Machine identity (FLY_MACHINE_ID / HA_MACHINE_ID / /etc/machine-id). Process generation is boot_id.';


--
-- Name: COLUMN spec_generation_locks.boot_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.spec_generation_locks.boot_id IS 'pid:process-start-ns on the holding process. Same machine + dead process → reclaim.';


--
-- Name: spec_generation_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.spec_generation_logs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid,
    model_id text NOT NULL,
    run_key text,
    success boolean,
    log text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    state text,
    message text,
    host_id text,
    boot_id text,
    finished_at timestamp with time zone,
    kind text,
    agent_cli text,
    agent_model text,
    error_kind text,
    engine_gaps jsonb,
    record_id uuid,
    CONSTRAINT spec_generation_logs_error_kind_check CHECK (((error_kind IS NULL) OR (error_kind = ANY (ARRAY['spec_invalid'::text, 'spec_bundle_incomplete'::text, 'spec_needs_declaration'::text, 'bundle_invalid'::text, 'blocked_by_engine'::text, 'analysis_refusal_unattributed'::text, 'agent_unavailable'::text, 'timed_out'::text, 'killed'::text, 'context_exhausted'::text]))))
);


--
-- Name: COLUMN spec_generation_logs.record_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.spec_generation_logs.record_id IS 'The spec_generations row this run wrote (set by the run). NULL when the run wrote none.';


--
-- Name: spec_generation_reviews; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.spec_generation_reviews (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    record_id uuid NOT NULL,
    model_id text NOT NULL,
    reviewer_id uuid,
    reviewer_role text NOT NULL,
    verdict text NOT NULL,
    report text,
    agent_cli text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    bundle_fingerprint text,
    run_revision text,
    engine_version text,
    admission jsonb,
    log_id uuid,
    CONSTRAINT spec_generation_reviews_reviewer_role_check CHECK ((reviewer_role = ANY (ARRAY['user'::text, 'admin'::text]))),
    CONSTRAINT spec_generation_reviews_verdict_check CHECK ((verdict = ANY (ARRAY['pass'::text, 'no_pass'::text, 'error'::text])))
);


--
-- Name: COLUMN spec_generation_reviews.log_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.spec_generation_reviews.log_id IS 'The spec_generation_logs row of the run that wrote this review. NULL before migration 0020.';


--
-- Name: spec_generations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.spec_generations (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid,
    model_id text NOT NULL,
    spec_yaml text NOT NULL,
    snapshot_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    new_component_yamls jsonb DEFAULT '[]'::jsonb NOT NULL,
    agent_cli text,
    status text DEFAULT 'generated'::text NOT NULL,
    contributed_at timestamp with time zone,
    pr_url text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    agent_model text,
    pending_detail jsonb,
    blocked_engine_version text,
    run_started_at timestamp with time zone,
    revision bigint DEFAULT 0 NOT NULL,
    CONSTRAINT spec_generations_status_check CHECK ((status = ANY (ARRAY['generated'::text, 'contributed'::text, 'merged'::text, 'rejected'::text, 'closed'::text, 'blocked_by_engine'::text, 'needs_repair'::text])))
);


--
-- Name: sweep_cache; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sweep_cache (
    key text NOT NULL,
    payload jsonb NOT NULL,
    hit_count integer DEFAULT 0 NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    last_used_at timestamp with time zone DEFAULT now()
);


--
-- Name: user_agent_credentials; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_agent_credentials (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    agent_cli text NOT NULL,
    api_key_enc text NOT NULL,
    base_url text,
    is_default boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    model text,
    task_models jsonb,
    name text DEFAULT ''::text NOT NULL,
    CONSTRAINT user_agent_credentials_agent_cli_check CHECK ((agent_cli = ANY (ARRAY['claude'::text, 'codex'::text, 'kimi'::text])))
);


--
-- Name: user_models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_models (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    name text NOT NULL,
    slug text NOT NULL,
    spec_json jsonb NOT NULL,
    is_public boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: user_module_variants; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_module_variants (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    component text NOT NULL,
    name text NOT NULL,
    dims jsonb NOT NULL,
    base_model_id text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT user_module_variants_dims_check CHECK ((jsonb_typeof(dims) = 'object'::text))
);


--
-- Name: admins admins_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admins
    ADD CONSTRAINT admins_pkey PRIMARY KEY (user_id);


--
-- Name: agent_task_defaults agent_task_defaults_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_task_defaults
    ADD CONSTRAINT agent_task_defaults_pkey PRIMARY KEY (agent_cli, task);


--
-- Name: analyze_logs analyze_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.analyze_logs
    ADD CONSTRAINT analyze_logs_pkey PRIMARY KEY (id);


--
-- Name: bench_lake_lease bench_lake_lease_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bench_lake_lease
    ADD CONSTRAINT bench_lake_lease_pkey PRIMARY KEY (id);


--
-- Name: cached_specs cached_specs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cached_specs
    ADD CONSTRAINT cached_specs_pkey PRIMARY KEY (model_id);


--
-- Name: login_logs login_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.login_logs
    ADD CONSTRAINT login_logs_pkey PRIMARY KEY (id);


--
-- Name: personal_access_tokens personal_access_tokens_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personal_access_tokens
    ADD CONSTRAINT personal_access_tokens_pkey PRIMARY KEY (id);


--
-- Name: personal_access_tokens personal_access_tokens_token_hash_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personal_access_tokens
    ADD CONSTRAINT personal_access_tokens_token_hash_key UNIQUE (token_hash);


--
-- Name: profiles profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.profiles
    ADD CONSTRAINT profiles_pkey PRIMARY KEY (id);


--
-- Name: spec_generation_locks spec_generation_locks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spec_generation_locks
    ADD CONSTRAINT spec_generation_locks_pkey PRIMARY KEY (model_id);


--
-- Name: spec_generation_logs spec_generation_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spec_generation_logs
    ADD CONSTRAINT spec_generation_logs_pkey PRIMARY KEY (id);


--
-- Name: spec_generation_reviews spec_generation_reviews_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spec_generation_reviews
    ADD CONSTRAINT spec_generation_reviews_pkey PRIMARY KEY (id);


--
-- Name: spec_generations spec_generations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spec_generations
    ADD CONSTRAINT spec_generations_pkey PRIMARY KEY (id);


--
-- Name: sweep_cache sweep_cache_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sweep_cache
    ADD CONSTRAINT sweep_cache_pkey PRIMARY KEY (key);


--
-- Name: user_agent_credentials user_agent_credentials_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_agent_credentials
    ADD CONSTRAINT user_agent_credentials_pkey PRIMARY KEY (id);


--
-- Name: user_agent_credentials user_agent_credentials_user_id_agent_cli_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_agent_credentials
    ADD CONSTRAINT user_agent_credentials_user_id_agent_cli_name_key UNIQUE (user_id, agent_cli, name);


--
-- Name: user_models user_models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_models
    ADD CONSTRAINT user_models_pkey PRIMARY KEY (id);


--
-- Name: user_models user_models_user_id_slug_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_models
    ADD CONSTRAINT user_models_user_id_slug_key UNIQUE (user_id, slug);


--
-- Name: user_module_variants user_module_variants_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_module_variants
    ADD CONSTRAINT user_module_variants_pkey PRIMARY KEY (id);


--
-- Name: user_module_variants user_module_variants_user_id_component_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_module_variants
    ADD CONSTRAINT user_module_variants_user_id_component_name_key UNIQUE (user_id, component, name);


--
-- Name: idx_analyze_logs_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_analyze_logs_created_at ON public.analyze_logs USING btree (created_at DESC);


--
-- Name: idx_analyze_logs_endpoint; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_analyze_logs_endpoint ON public.analyze_logs USING btree (endpoint);


--
-- Name: idx_analyze_logs_model_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_analyze_logs_model_id ON public.analyze_logs USING btree (model_id);


--
-- Name: idx_cached_specs_hits; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cached_specs_hits ON public.cached_specs USING btree (hit_count DESC);


--
-- Name: idx_cached_specs_last_used; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cached_specs_last_used ON public.cached_specs USING btree (last_used_at DESC);


--
-- Name: idx_login_logs_login_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_login_logs_login_at ON public.login_logs USING btree (login_at DESC);


--
-- Name: idx_login_logs_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_login_logs_user_id ON public.login_logs USING btree (user_id);


--
-- Name: idx_pat_hash_live; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pat_hash_live ON public.personal_access_tokens USING btree (token_hash) WHERE (revoked_at IS NULL);


--
-- Name: idx_pat_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pat_user ON public.personal_access_tokens USING btree (user_id);


--
-- Name: idx_spec_generation_logs_error_kind_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generation_logs_error_kind_created ON public.spec_generation_logs USING btree (error_kind, created_at DESC) WHERE (error_kind IS NOT NULL);


--
-- Name: idx_spec_generation_logs_one_running; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_spec_generation_logs_one_running ON public.spec_generation_logs USING btree (run_key) WHERE ((state = 'running'::text) AND (run_key IS NOT NULL));


--
-- Name: idx_spec_generation_logs_run_key_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generation_logs_run_key_created ON public.spec_generation_logs USING btree (run_key, created_at DESC);


--
-- Name: idx_spec_generation_logs_running; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generation_logs_running ON public.spec_generation_logs USING btree (state) WHERE (state = 'running'::text);


--
-- Name: idx_spec_generation_logs_user_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generation_logs_user_created ON public.spec_generation_logs USING btree (user_id, created_at DESC);


--
-- Name: idx_spec_generation_logs_user_model; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generation_logs_user_model ON public.spec_generation_logs USING btree (user_id, model_id, created_at DESC);


--
-- Name: idx_spec_generation_reviews_record; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generation_reviews_record ON public.spec_generation_reviews USING btree (record_id, created_at DESC);


--
-- Name: idx_spec_generation_reviews_reviewer; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generation_reviews_reviewer ON public.spec_generation_reviews USING btree (reviewer_id, created_at DESC);


--
-- Name: idx_spec_generations_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generations_created ON public.spec_generations USING btree (created_at DESC);


--
-- Name: idx_spec_generations_model; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generations_model ON public.spec_generations USING btree (model_id);


--
-- Name: idx_spec_generations_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generations_user ON public.spec_generations USING btree (user_id, created_at DESC);


--
-- Name: idx_spec_generations_user_model_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_generations_user_model_created ON public.spec_generations USING btree (user_id, model_id, created_at DESC);


--
-- Name: idx_spec_locks_expires_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_spec_locks_expires_at ON public.spec_generation_locks USING btree (expires_at);


--
-- Name: idx_sweep_cache_last_used; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sweep_cache_last_used ON public.sweep_cache USING btree (last_used_at DESC);


--
-- Name: idx_user_agent_credentials_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_user_agent_credentials_user ON public.user_agent_credentials USING btree (user_id);


--
-- Name: idx_user_models_public; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_user_models_public ON public.user_models USING btree (is_public) WHERE (is_public = true);


--
-- Name: idx_user_models_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_user_models_user_id ON public.user_models USING btree (user_id);


--
-- Name: spec_generations_one_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX spec_generations_one_pending ON public.spec_generations USING btree (user_id, model_id) WHERE (status = ANY (ARRAY['blocked_by_engine'::text, 'needs_repair'::text]));


--
-- Name: uniq_user_default_agent; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uniq_user_default_agent ON public.user_agent_credentials USING btree (user_id) WHERE is_default;


--
-- Name: user_module_variants module_variants_touch_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER module_variants_touch_updated_at BEFORE UPDATE ON public.user_module_variants FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: spec_generations spec_generations_bump_revision; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER spec_generations_bump_revision BEFORE UPDATE ON public.spec_generations FOR EACH ROW EXECUTE FUNCTION public.spec_generations_bump_revision();


--
-- Name: profiles update_profiles_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON public.profiles FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: user_models update_user_models_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_user_models_updated_at BEFORE UPDATE ON public.user_models FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: admins admins_created_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admins
    ADD CONSTRAINT admins_created_by_fkey FOREIGN KEY (created_by) REFERENCES auth.users(id);


--
-- Name: admins admins_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admins
    ADD CONSTRAINT admins_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;


--
-- Name: analyze_logs analyze_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.analyze_logs
    ADD CONSTRAINT analyze_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE SET NULL;


--
-- Name: cached_specs cached_specs_generated_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cached_specs
    ADD CONSTRAINT cached_specs_generated_by_fkey FOREIGN KEY (generated_by) REFERENCES auth.users(id) ON DELETE SET NULL;


--
-- Name: login_logs login_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.login_logs
    ADD CONSTRAINT login_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;


--
-- Name: personal_access_tokens personal_access_tokens_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personal_access_tokens
    ADD CONSTRAINT personal_access_tokens_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;


--
-- Name: profiles profiles_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.profiles
    ADD CONSTRAINT profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id) ON DELETE CASCADE;


--
-- Name: spec_generation_logs spec_generation_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spec_generation_logs
    ADD CONSTRAINT spec_generation_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE SET NULL;


--
-- Name: spec_generation_reviews spec_generation_reviews_record_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spec_generation_reviews
    ADD CONSTRAINT spec_generation_reviews_record_id_fkey FOREIGN KEY (record_id) REFERENCES public.spec_generations(id) ON DELETE CASCADE;


--
-- Name: spec_generation_reviews spec_generation_reviews_reviewer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spec_generation_reviews
    ADD CONSTRAINT spec_generation_reviews_reviewer_id_fkey FOREIGN KEY (reviewer_id) REFERENCES auth.users(id) ON DELETE SET NULL;


--
-- Name: spec_generations spec_generations_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spec_generations
    ADD CONSTRAINT spec_generations_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE SET NULL;


--
-- Name: user_agent_credentials user_agent_credentials_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_agent_credentials
    ADD CONSTRAINT user_agent_credentials_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;


--
-- Name: user_models user_models_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_models
    ADD CONSTRAINT user_models_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;


--
-- Name: user_module_variants user_module_variants_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_module_variants
    ADD CONSTRAINT user_module_variants_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;


--
-- Name: admins Admins can delete admins; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Admins can delete admins" ON public.admins FOR DELETE USING ((EXISTS ( SELECT 1
   FROM public.admins a
  WHERE (a.user_id = auth.uid()))));


--
-- Name: admins Admins can insert admins; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Admins can insert admins" ON public.admins FOR INSERT WITH CHECK ((EXISTS ( SELECT 1
   FROM public.admins a
  WHERE (a.user_id = auth.uid()))));


--
-- Name: admins Admins can view all admins; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Admins can view all admins" ON public.admins FOR SELECT USING ((EXISTS ( SELECT 1
   FROM public.admins a
  WHERE (a.user_id = auth.uid()))));


--
-- Name: login_logs Admins can view all login logs; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Admins can view all login logs" ON public.login_logs FOR SELECT USING (public.is_admin(auth.uid()));


--
-- Name: user_models Admins can view all models; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Admins can view all models" ON public.user_models FOR SELECT USING (public.is_admin(auth.uid()));


--
-- Name: user_module_variants Admins can view all module variants; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Admins can view all module variants" ON public.user_module_variants FOR SELECT USING (public.is_admin(auth.uid()));


--
-- Name: analyze_logs Admins can view analyze logs; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Admins can view analyze logs" ON public.analyze_logs FOR SELECT USING (public.is_admin(auth.uid()));


--
-- Name: analyze_logs Anyone can insert analyze logs; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Anyone can insert analyze logs" ON public.analyze_logs FOR INSERT WITH CHECK (true);


--
-- Name: user_models Public models are viewable by everyone; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Public models are viewable by everyone" ON public.user_models FOR SELECT USING ((is_public = true));


--
-- Name: profiles Public profiles are viewable by everyone; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Public profiles are viewable by everyone" ON public.profiles FOR SELECT USING (true);


--
-- Name: login_logs System can insert login logs; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "System can insert login logs" ON public.login_logs FOR INSERT WITH CHECK (true);


--
-- Name: user_models Users can delete own models; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can delete own models" ON public.user_models FOR DELETE USING ((auth.uid() = user_id));


--
-- Name: profiles Users can delete own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can delete own profile" ON public.profiles FOR DELETE USING ((auth.uid() = id));


--
-- Name: user_models Users can insert own models; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can insert own models" ON public.user_models FOR INSERT WITH CHECK ((auth.uid() = user_id));


--
-- Name: profiles Users can insert own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can insert own profile" ON public.profiles FOR INSERT WITH CHECK ((auth.uid() = id));


--
-- Name: user_models Users can update own models; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can update own models" ON public.user_models FOR UPDATE USING ((auth.uid() = user_id)) WITH CHECK ((auth.uid() = user_id));


--
-- Name: profiles Users can update own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can update own profile" ON public.profiles FOR UPDATE USING ((auth.uid() = id)) WITH CHECK ((auth.uid() = id));


--
-- Name: login_logs Users can view own login logs; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can view own login logs" ON public.login_logs FOR SELECT USING ((auth.uid() = user_id));


--
-- Name: user_models Users can view own models; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can view own models" ON public.user_models FOR SELECT USING ((auth.uid() = user_id));


--
-- Name: user_module_variants Users can view own module variants; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can view own module variants" ON public.user_module_variants FOR SELECT USING ((auth.uid() = user_id));


--
-- Name: admins; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.admins ENABLE ROW LEVEL SECURITY;

--
-- Name: agent_task_defaults; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.agent_task_defaults ENABLE ROW LEVEL SECURITY;

--
-- Name: analyze_logs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.analyze_logs ENABLE ROW LEVEL SECURITY;

--
-- Name: bench_lake_lease; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.bench_lake_lease ENABLE ROW LEVEL SECURITY;

--
-- Name: cached_specs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.cached_specs ENABLE ROW LEVEL SECURITY;

--
-- Name: login_logs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.login_logs ENABLE ROW LEVEL SECURITY;

--
-- Name: personal_access_tokens; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.personal_access_tokens ENABLE ROW LEVEL SECURITY;

--
-- Name: profiles; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

--
-- Name: spec_generation_locks; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.spec_generation_locks ENABLE ROW LEVEL SECURITY;

--
-- Name: spec_generation_logs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.spec_generation_logs ENABLE ROW LEVEL SECURITY;

--
-- Name: spec_generation_logs spec_generation_logs_owner_read; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY spec_generation_logs_owner_read ON public.spec_generation_logs FOR SELECT USING ((auth.uid() = user_id));


--
-- Name: spec_generation_reviews; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.spec_generation_reviews ENABLE ROW LEVEL SECURITY;

--
-- Name: spec_generation_reviews spec_generation_reviews_owner_read; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY spec_generation_reviews_owner_read ON public.spec_generation_reviews FOR SELECT USING ((EXISTS ( SELECT 1
   FROM public.spec_generations sg
  WHERE ((sg.id = spec_generation_reviews.record_id) AND (sg.user_id = auth.uid())))));


--
-- Name: spec_generations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.spec_generations ENABLE ROW LEVEL SECURITY;

--
-- Name: spec_generations spec_generations owner read; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "spec_generations owner read" ON public.spec_generations FOR SELECT USING ((auth.uid() = user_id));


--
-- Name: sweep_cache; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.sweep_cache ENABLE ROW LEVEL SECURITY;

--
-- Name: user_agent_credentials; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.user_agent_credentials ENABLE ROW LEVEL SECURITY;

--
-- Name: user_models; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.user_models ENABLE ROW LEVEL SECURITY;

--
-- Name: user_module_variants; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.user_module_variants ENABLE ROW LEVEL SECURITY;

--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: -
--

GRANT USAGE ON SCHEMA public TO postgres;
GRANT USAGE ON SCHEMA public TO anon;
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT USAGE ON SCHEMA public TO service_role;


--
-- Name: FUNCTION add_admin(p_user_id uuid); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.add_admin(p_user_id uuid) TO authenticated;


--
-- Name: FUNCTION bench_lake_claim(p_lake uuid, p_machine text); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.bench_lake_claim(p_lake uuid, p_machine text) FROM PUBLIC;
GRANT ALL ON FUNCTION public.bench_lake_claim(p_lake uuid, p_machine text) TO service_role;


--
-- Name: FUNCTION bench_lake_transfer(p_old text, p_new text); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.bench_lake_transfer(p_old text, p_new text) FROM PUBLIC;


--
-- Name: FUNCTION bump_cached_spec_hit(p_model_id text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.bump_cached_spec_hit(p_model_id text) TO service_role;


--
-- Name: FUNCTION cleanup_stale_spec_locks(); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.cleanup_stale_spec_locks() TO service_role;


--
-- Name: FUNCTION create_user_profile_if_not_exists(p_user_id uuid, p_username text, p_avatar_url text, p_full_name text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.create_user_profile_if_not_exists(p_user_id uuid, p_username text, p_avatar_url text, p_full_name text) TO authenticated;
GRANT ALL ON FUNCTION public.create_user_profile_if_not_exists(p_user_id uuid, p_username text, p_avatar_url text, p_full_name text) TO anon;


--
-- Name: FUNCTION delete_cached_spec(p_model_id text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.delete_cached_spec(p_model_id text) TO service_role;


--
-- Name: FUNCTION delete_user_data(p_user_id uuid); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.delete_user_data(p_user_id uuid) TO authenticated;


--
-- Name: FUNCTION get_active_users_stats(p_start_date date, p_end_date date); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_active_users_stats(p_start_date date, p_end_date date) TO authenticated;


--
-- Name: FUNCTION get_active_users_today(); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_active_users_today() TO authenticated;


--
-- Name: FUNCTION get_cached_spec(p_model_id text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_cached_spec(p_model_id text) TO service_role;


--
-- Name: FUNCTION get_cached_spec_detail(p_model_id text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_cached_spec_detail(p_model_id text) TO service_role;


--
-- Name: FUNCTION get_cached_specs(p_limit integer, p_offset integer, p_sort text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_cached_specs(p_limit integer, p_offset integer, p_sort text) TO service_role;


--
-- Name: FUNCTION get_cached_specs_count(); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_cached_specs_count() TO service_role;


--
-- Name: FUNCTION get_daily_call_trend(p_start_date date, p_end_date date); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_daily_call_trend(p_start_date date, p_end_date date) TO authenticated;


--
-- Name: FUNCTION get_inference_query_dist(p_start_date date, p_end_date date); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_inference_query_dist(p_start_date date, p_end_date date) TO authenticated;


--
-- Name: FUNCTION get_latency_summary(p_start_date date, p_end_date date); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_latency_summary(p_start_date date, p_end_date date) TO authenticated;


--
-- Name: FUNCTION get_login_logs_with_email(p_limit integer, p_offset integer, p_start_date date, p_end_date date, p_user_email text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_login_logs_with_email(p_limit integer, p_offset integer, p_start_date date, p_end_date date, p_user_email text) TO authenticated;


--
-- Name: FUNCTION get_spec_generation_detail(p_id uuid); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.get_spec_generation_detail(p_id uuid) FROM PUBLIC;
GRANT ALL ON FUNCTION public.get_spec_generation_detail(p_id uuid) TO authenticated;
GRANT ALL ON FUNCTION public.get_spec_generation_detail(p_id uuid) TO service_role;


--
-- Name: FUNCTION get_spec_generation_reviews(p_record_id uuid); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_spec_generation_reviews(p_record_id uuid) TO service_role;


--
-- Name: FUNCTION get_spec_generation_runs(p_limit integer, p_offset integer, p_error_kind text); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.get_spec_generation_runs(p_limit integer, p_offset integer, p_error_kind text) FROM PUBLIC;
GRANT ALL ON FUNCTION public.get_spec_generation_runs(p_limit integer, p_offset integer, p_error_kind text) TO authenticated;
GRANT ALL ON FUNCTION public.get_spec_generation_runs(p_limit integer, p_offset integer, p_error_kind text) TO service_role;


--
-- Name: FUNCTION get_spec_generation_runs_count(p_error_kind text); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.get_spec_generation_runs_count(p_error_kind text) FROM PUBLIC;
GRANT ALL ON FUNCTION public.get_spec_generation_runs_count(p_error_kind text) TO authenticated;
GRANT ALL ON FUNCTION public.get_spec_generation_runs_count(p_error_kind text) TO service_role;


--
-- Name: FUNCTION get_spec_generations(p_limit integer, p_offset integer, p_sort text, p_status text); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.get_spec_generations(p_limit integer, p_offset integer, p_sort text, p_status text) FROM PUBLIC;
GRANT ALL ON FUNCTION public.get_spec_generations(p_limit integer, p_offset integer, p_sort text, p_status text) TO authenticated;
GRANT ALL ON FUNCTION public.get_spec_generations(p_limit integer, p_offset integer, p_sort text, p_status text) TO service_role;


--
-- Name: FUNCTION get_spec_generations_count(p_status text); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.get_spec_generations_count(p_status text) FROM PUBLIC;
GRANT ALL ON FUNCTION public.get_spec_generations_count(p_status text) TO authenticated;
GRANT ALL ON FUNCTION public.get_spec_generations_count(p_status text) TO service_role;


--
-- Name: FUNCTION get_top_models(p_start_date date, p_end_date date, p_limit integer); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_top_models(p_start_date date, p_end_date date, p_limit integer) TO authenticated;


--
-- Name: FUNCTION get_total_analyze_logs_count(); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_total_analyze_logs_count() TO authenticated;


--
-- Name: FUNCTION get_total_login_logs_count(); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_total_login_logs_count() TO authenticated;


--
-- Name: FUNCTION get_user_analyze_logs(p_user_id uuid, p_limit integer, p_offset integer); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_user_analyze_logs(p_user_id uuid, p_limit integer, p_offset integer) TO authenticated;


--
-- Name: FUNCTION get_user_summary(p_user_id uuid); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_user_summary(p_user_id uuid) TO authenticated;


--
-- Name: FUNCTION get_users_count(); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_users_count() TO authenticated;


--
-- Name: FUNCTION get_users_with_admin_status(p_limit integer, p_offset integer); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.get_users_with_admin_status(p_limit integer, p_offset integer) TO authenticated;


--
-- Name: FUNCTION is_admin(check_user_id uuid); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.is_admin(check_user_id uuid) TO authenticated;
GRANT ALL ON FUNCTION public.is_admin(check_user_id uuid) TO anon;


--
-- Name: FUNCTION is_super_admin(check_user_id uuid); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.is_super_admin(check_user_id uuid) TO authenticated;


--
-- Name: FUNCTION log_user_login(p_user_id uuid, p_ip_address inet, p_user_agent text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.log_user_login(p_user_id uuid, p_ip_address inet, p_user_agent text) TO authenticated;


--
-- Name: FUNCTION record_generation(p_user_id uuid, p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_run_started_at timestamp with time zone); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.record_generation(p_user_id uuid, p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_run_started_at timestamp with time zone) FROM PUBLIC;
GRANT ALL ON FUNCTION public.record_generation(p_user_id uuid, p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_run_started_at timestamp with time zone) TO service_role;


--
-- Name: FUNCTION record_pending_generation(p_user_id uuid, p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_engine_gaps jsonb, p_engine_version text, p_run_started_at timestamp with time zone); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.record_pending_generation(p_user_id uuid, p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_engine_gaps jsonb, p_engine_version text, p_run_started_at timestamp with time zone) FROM PUBLIC;
GRANT ALL ON FUNCTION public.record_pending_generation(p_user_id uuid, p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_engine_gaps jsonb, p_engine_version text, p_run_started_at timestamp with time zone) TO service_role;


--
-- Name: FUNCTION release_spec_lock(p_model_id text, p_host_id text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.release_spec_lock(p_model_id text, p_host_id text) TO service_role;


--
-- Name: FUNCTION release_spec_lock_v2(p_model_id text, p_host_id text, p_boot_id text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.release_spec_lock_v2(p_model_id text, p_host_id text, p_boot_id text) TO service_role;


--
-- Name: FUNCTION remove_admin(p_user_id uuid); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.remove_admin(p_user_id uuid) TO authenticated;


--
-- Name: FUNCTION set_super_admin(p_user_id uuid, p_is_super boolean); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.set_super_admin(p_user_id uuid, p_is_super boolean) TO authenticated;


--
-- Name: FUNCTION spec_generations_bump_revision(); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.spec_generations_bump_revision() FROM PUBLIC;


--
-- Name: FUNCTION sweep_cache_clear(); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.sweep_cache_clear() TO service_role;


--
-- Name: FUNCTION sweep_cache_get(p_key text); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.sweep_cache_get(p_key text) TO service_role;


--
-- Name: FUNCTION sweep_cache_upsert(p_key text, p_payload jsonb); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.sweep_cache_upsert(p_key text, p_payload jsonb) TO service_role;


--
-- Name: FUNCTION transition_pending_generation(p_id uuid, p_expected_revision bigint, p_to text, p_engine_gaps jsonb, p_repairable_errors jsonb, p_engine_version text, p_spec_yaml text, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_run_started_at timestamp with time zone); Type: ACL; Schema: public; Owner: -
--

REVOKE ALL ON FUNCTION public.transition_pending_generation(p_id uuid, p_expected_revision bigint, p_to text, p_engine_gaps jsonb, p_repairable_errors jsonb, p_engine_version text, p_spec_yaml text, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_run_started_at timestamp with time zone) FROM PUBLIC;
GRANT ALL ON FUNCTION public.transition_pending_generation(p_id uuid, p_expected_revision bigint, p_to text, p_engine_gaps jsonb, p_repairable_errors jsonb, p_engine_version text, p_spec_yaml text, p_new_component_yamls jsonb, p_agent_cli text, p_agent_model text, p_run_started_at timestamp with time zone) TO service_role;


--
-- Name: FUNCTION try_acquire_spec_lock(p_model_id text, p_host_id text, p_ttl_seconds integer); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.try_acquire_spec_lock(p_model_id text, p_host_id text, p_ttl_seconds integer) TO service_role;


--
-- Name: FUNCTION try_acquire_spec_lock_v2(p_model_id text, p_host_id text, p_boot_id text, p_ttl_seconds integer); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.try_acquire_spec_lock_v2(p_model_id text, p_host_id text, p_boot_id text, p_ttl_seconds integer) TO service_role;


--
-- Name: FUNCTION upsert_cached_spec(p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_generated_by uuid); Type: ACL; Schema: public; Owner: -
--

GRANT ALL ON FUNCTION public.upsert_cached_spec(p_model_id text, p_spec_yaml text, p_snapshot_json jsonb, p_new_component_yamls jsonb, p_agent_cli text, p_generated_by uuid) TO service_role;


--
-- Name: TABLE admins; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.admins TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.admins TO authenticated;
GRANT ALL ON TABLE public.admins TO service_role;


--
-- Name: TABLE agent_task_defaults; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.agent_task_defaults TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.agent_task_defaults TO authenticated;
GRANT ALL ON TABLE public.agent_task_defaults TO service_role;


--
-- Name: TABLE analyze_logs; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.analyze_logs TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.analyze_logs TO authenticated;
GRANT ALL ON TABLE public.analyze_logs TO service_role;


--
-- Name: TABLE bench_lake_lease; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT ON TABLE public.bench_lake_lease TO service_role;


--
-- Name: TABLE cached_specs; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.cached_specs TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.cached_specs TO authenticated;
GRANT ALL ON TABLE public.cached_specs TO service_role;


--
-- Name: TABLE login_logs; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.login_logs TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.login_logs TO authenticated;
GRANT ALL ON TABLE public.login_logs TO service_role;


--
-- Name: TABLE personal_access_tokens; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.personal_access_tokens TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.personal_access_tokens TO authenticated;
GRANT ALL ON TABLE public.personal_access_tokens TO service_role;


--
-- Name: TABLE profiles; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.profiles TO anon;
GRANT SELECT,REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.profiles TO authenticated;
GRANT ALL ON TABLE public.profiles TO service_role;


--
-- Name: TABLE spec_generation_locks; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.spec_generation_locks TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.spec_generation_locks TO authenticated;
GRANT ALL ON TABLE public.spec_generation_locks TO service_role;


--
-- Name: TABLE spec_generation_logs; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.spec_generation_logs TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.spec_generation_logs TO authenticated;
GRANT ALL ON TABLE public.spec_generation_logs TO service_role;


--
-- Name: TABLE spec_generation_reviews; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.spec_generation_reviews TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.spec_generation_reviews TO authenticated;
GRANT ALL ON TABLE public.spec_generation_reviews TO service_role;


--
-- Name: TABLE spec_generations; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.spec_generations TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.spec_generations TO authenticated;
GRANT ALL ON TABLE public.spec_generations TO service_role;


--
-- Name: TABLE sweep_cache; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.sweep_cache TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.sweep_cache TO authenticated;
GRANT ALL ON TABLE public.sweep_cache TO service_role;


--
-- Name: TABLE user_agent_credentials; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.user_agent_credentials TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.user_agent_credentials TO authenticated;
GRANT ALL ON TABLE public.user_agent_credentials TO service_role;


--
-- Name: TABLE user_models; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.user_models TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.user_models TO authenticated;
GRANT ALL ON TABLE public.user_models TO service_role;


--
-- Name: TABLE user_module_variants; Type: ACL; Schema: public; Owner: -
--

GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.user_module_variants TO anon;
GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLE public.user_module_variants TO authenticated;
GRANT ALL ON TABLE public.user_module_variants TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT UPDATE ON SEQUENCES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT UPDATE ON SEQUENCES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON SEQUENCES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON SEQUENCES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON SEQUENCES TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR FUNCTIONS; Type: DEFAULT ACL; Schema: public; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON FUNCTIONS TO postgres;


--
-- Name: DEFAULT PRIVILEGES FOR FUNCTIONS; Type: DEFAULT ACL; Schema: public; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON FUNCTIONS TO postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON FUNCTIONS TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON FUNCTIONS TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON FUNCTIONS TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON TABLES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public GRANT ALL ON TABLES TO service_role;


--
-- PostgreSQL database dump complete
--

